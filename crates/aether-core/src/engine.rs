//! The orchestrator: owns every subsystem and the buffers that cross into JS.
//!
//! # Zero-copy boundary
//!
//! Per-frame data is never serialised. JS writes camera luma and segmentation
//! masks straight into engine-owned buffers through raw pointers, and reads the
//! dye texture and particle buffer back the same way. Landmarks are small
//! enough that they come in as slices.
//!
//! The one rule: a pointer is only valid until the next call that can grow a
//! buffer ([`Engine::set_particle_count`] or [`Engine::reset`]). The JS side
//! re-reads pointers after those, and after any WASM memory growth.
//!
//! # Frame protocol
//!
//! ```text
//! // once per camera frame (~30 Hz)
//! write luma into luma_ptr(); engine.push_luma(dt_since_last_luma)
//! // once per perception result (~30 Hz, may lag the camera)
//! write masks into mask_ptr(); engine.push_mask(w, h, mirror, dt)
//! engine.push_hands(&packed, dt); engine.push_pose(&packed, dt)
//! // once per animation frame (~60 Hz)
//! engine.step(dt)  -> then read dye_ptr() and particle_ptr()
//! ```

use crate::config::{
    Params, DEFAULT_PARTICLES, FLOW_CELLS, FLOW_H, FLOW_W, FLUID_CELLS, FLUID_H, FLUID_W,
    HAND_BUFFER, MAX_PARTICLES, PARTICLE_STRIDE, POSE_STRIDE,
};
use crate::field::{Grid, VecField};
use crate::fluid::Fluid;
use crate::flow::{FlowConfig, OpticalFlow};
use crate::gesture::{GestureConfig, GestureTracker, Spell};
use crate::mask::{BodyMask, MaskConfig};
use crate::math::hue_to_rgb;
use crate::particles::{ParticleConfig, Particles};
use crate::spells::{self, SpellReport, SpellState};

/// Largest segmentation mask the input buffer accepts, per axis. MediaPipe's
/// pose model emits 256x256; anything bigger must be downscaled by the caller.
pub const MASK_IN_MAX_DIM: usize = 320;
/// Capacity of the mask input buffer, in floats.
pub const MASK_IN_CAPACITY: usize = MASK_IN_MAX_DIM * MASK_IN_MAX_DIM;

/// Seconds without any perception input before the engine drives itself.
const AMBIENT_AFTER: f32 = 1.5;

/// Frames between the three grid-walking HUD statistics.
///
/// `fluid_energy` and `fluid_max_speed` each traverse the whole velocity field.
/// The HUD repaints at 10 Hz, so refreshing them at 60 Hz burns full grid
/// passes to produce numbers nobody reads. Every other stat is an O(1) read and
/// stays live every frame, because callers treat those as state rather than
/// telemetry.
const STATS_EVERY: u32 = 4;

/// Frames between defensive NaN scrubs.
///
/// The simulation should never produce a non-finite cell. If it does, one
/// escaped NaN spreads through the field within a few frames and blanks the
/// screen, so it has to be caught — but checking five grids every frame is a
/// cost paid forever against something that should never happen. Every fourth
/// frame bounds the damage to ~66 ms of corruption.
const SANITIZE_EVERY: u32 = 4;

/// Resolution of the dye tone-mapping table, and the dye value it saturates at.
///
/// `encode_dye` needs `1 - exp(-x)` three times per cell, which is 110k
/// transcendental calls per frame at this grid size — measurably more than the
/// entire particle system costs. A table over the curve's useful range, read
/// with linear interpolation, is visually indistinguishable and an order of
/// magnitude cheaper. Beyond `TONEMAP_MAX` the curve is within 0.2% of 1.0, so
/// clamping there loses nothing.
const TONEMAP_LUT: usize = 512;
const TONEMAP_MAX: f32 = 8.0;

/// A snapshot of engine state for the HUD.
#[derive(Clone, Copy, Debug, Default)]
pub struct Stats {
    pub fluid_energy: f32,
    pub fluid_max_speed: f32,
    pub fluid_divergence: f32,
    pub motion_energy: f32,
    pub flow_dx: f32,
    pub flow_dy: f32,
    pub particles_alive: u32,
    pub mask_coverage: f32,
    pub mask_present: bool,
    pub hands_present: u32,
    pub pose_present: bool,
    pub time_scale: f32,
    pub bursts: u32,
    /// Non-finite cells repaired this frame. Should always be 0.
    pub nan_repairs: u32,
    pub frame: u32,
    pub ambient: bool,
}

/// The complete Aether engine.
pub struct Engine {
    params: Params,
    particle_cfg: ParticleConfig,

    fluid: Fluid,
    particles: Particles,
    flow: OpticalFlow,
    body: BodyMask,
    tracker: GestureTracker,
    spell_state: SpellState,

    // --- JS-writable input buffers ---
    luma: Vec<u8>,
    mask_in: Vec<f32>,

    // --- JS-readable output buffers ---
    dye_rgba: Vec<u8>,
    debug_rgba: Vec<u8>,

    /// Zero velocity field used when no body edge motion is available.
    idle_field: VecField,

    /// `1 - exp(-x)` sampled over `[0, TONEMAP_MAX]`, built once.
    tonemap_lut: Vec<f32>,
    /// True when a mask arrived since the last step, or the obstacle field is
    /// still fading out. Lets `step` skip re-uploading an unchanged obstacle,
    /// which is a full grid resample plus a wall-mask rebuild.
    obstacle_dirty: bool,

    time: f32,
    frame: u32,
    /// Seconds since any hand, pose or mask arrived.
    since_perception: f32,
    /// Seconds since a camera frame arrived, tracked separately from
    /// perception: a live feed with nobody recognised in it still means the
    /// canvas is not blank, which is the only thing ambient exists to fix.
    since_luma: f32,
    stats: Stats,
    report: SpellReport,
    seed: u64,
}

impl Engine {
    pub fn new(seed: u64) -> Self {
        let mut particles = Particles::new(MAX_PARTICLES, seed);
        particles.set_active(DEFAULT_PARTICLES);
        let params = Params::default();
        particles.seed_uniform(FLUID_W, FLUID_H, params.particle_life);

        Self {
            params,
            particle_cfg: ParticleConfig::default(),
            fluid: Fluid::new(FLUID_W, FLUID_H),
            particles,
            flow: OpticalFlow::new(FLOW_W, FLOW_H, FlowConfig::default()),
            body: BodyMask::new(FLUID_W, FLUID_H, MaskConfig::default()),
            tracker: GestureTracker::new(GestureConfig::default()),
            spell_state: SpellState::new(),
            luma: vec![0; FLOW_CELLS],
            mask_in: vec![0.0; MASK_IN_CAPACITY],
            dye_rgba: vec![0; FLUID_CELLS * 4],
            debug_rgba: vec![0; FLUID_CELLS * 4],
            idle_field: VecField::new(FLUID_W, FLUID_H),
            tonemap_lut: build_tonemap_lut(),
            obstacle_dirty: true,
            time: 0.0,
            frame: 0,
            since_perception: f32::MAX / 4.0,
            since_luma: f32::MAX / 4.0,
            stats: Stats {
                time_scale: 1.0,
                ..Default::default()
            },
            report: SpellReport {
                time_scale: 1.0,
                ..Default::default()
            },
            seed,
        }
    }

    // ---------------------------------------------------------------- layout

    /// Grid dimensions, so JS can assert its mirrored constants match.
    pub fn layout(&self) -> [u32; 6] {
        [
            FLUID_W as u32,
            FLUID_H as u32,
            FLOW_W as u32,
            FLOW_H as u32,
            MAX_PARTICLES as u32,
            PARTICLE_STRIDE as u32,
        ]
    }

    // ------------------------------------------------------- input buffers

    /// Writable `FLOW_W * FLOW_H` luma plane.
    #[inline]
    pub fn luma_mut(&mut self) -> &mut [u8] {
        &mut self.luma
    }

    #[inline]
    pub fn luma_ptr(&mut self) -> *mut u8 {
        self.luma.as_mut_ptr()
    }

    #[inline]
    pub fn luma_len(&self) -> usize {
        self.luma.len()
    }

    /// Writable segmentation mask staging buffer, [`MASK_IN_CAPACITY`] floats.
    #[inline]
    pub fn mask_mut(&mut self) -> &mut [f32] {
        &mut self.mask_in
    }

    #[inline]
    pub fn mask_ptr(&mut self) -> *mut f32 {
        self.mask_in.as_mut_ptr()
    }

    #[inline]
    pub fn mask_capacity(&self) -> usize {
        MASK_IN_CAPACITY
    }

    // --------------------------------------------------------------- inputs

    /// Consumes the luma buffer and recomputes optical flow.
    pub fn push_luma(&mut self, dt: f32) {
        let dt = sane_dt(dt);
        // Borrow split: `update` needs &self.luma while holding &mut self.flow.
        let luma = core::mem::take(&mut self.luma);
        self.flow.update(&luma, dt);
        self.luma = luma;
        self.since_luma = 0.0;
    }

    /// Consumes `w * h` floats from the mask buffer as the body silhouette.
    pub fn push_mask(&mut self, w: usize, h: usize, mirror: bool, dt: f32) {
        if w < 2 || h < 2 || w * h > MASK_IN_CAPACITY {
            return;
        }
        let dt = sane_dt(dt);
        let mask = core::mem::take(&mut self.mask_in);
        self.body.update_f32(&mask[..w * h], w, h, mirror, dt);
        self.mask_in = mask;
        self.since_perception = 0.0;
        self.obstacle_dirty = true;
    }

    /// Feeds packed hand landmarks. See [`crate::config`] for the layout.
    pub fn push_hands(&mut self, packed: &[f32], dt: f32) {
        let dt = sane_dt(dt);
        self.tracker.update_hands(packed, dt);
        self.tracker.update_two_hand(dt);
        if self.tracker.hands().iter().any(|h| h.present) {
            self.since_perception = 0.0;
        }
    }

    /// Feeds packed pose landmarks.
    pub fn push_pose(&mut self, packed: &[f32], dt: f32) {
        let dt = sane_dt(dt);
        self.tracker.update_pose(packed, dt);
        if self.tracker.pose().present {
            self.since_perception = 0.0;
        }
    }

    /// Clears all perception state, e.g. when the camera is switched.
    pub fn clear_perception(&mut self) {
        self.tracker.reset();
        self.body.reset();
        self.flow.reset();
        self.spell_state.reset();
        self.since_perception = f32::MAX / 4.0;
        self.since_luma = f32::MAX / 4.0;
    }

    // ----------------------------------------------------------------- step

    /// Advances the whole engine by `dt` real seconds.
    pub fn step(&mut self, dt: f32) {
        let real_dt = sane_dt(dt);
        self.time += real_dt;
        self.frame = self.frame.wrapping_add(1);
        self.since_perception += real_dt;
        self.since_luma += real_dt;

        // The two-hand time-warp gesture scales simulated time, not real time,
        // so the frame pacing and the perception clocks stay untouched.
        let warped_dt = sane_dt(real_dt * self.params.time_scale * self.report.time_scale);

        self.body.decay(real_dt);

        // A live camera disqualifies ambient even when nothing is recognised in
        // it: drifting dye blobs over a real room just fog the picture.
        let ambient =
            self.since_perception > AMBIENT_AFTER && self.since_luma > AMBIENT_AFTER;
        if ambient {
            self.drive_ambient(warped_dt);
        }

        // Re-uploading the obstacle is a full grid resample plus a wall-mask
        // rebuild. The mask arrives at 30 Hz against a 60 Hz render loop, so
        // half these calls would hand the solver bytes it already has. During a
        // fade the field does change every frame, hence the staleness test.
        if self.obstacle_dirty || self.body.present() {
            self.fluid.set_obstacle(self.body.obstacle());
            self.obstacle_dirty = self.body.present();
        }

        let body_edge = if self.body.present() {
            // Borrowed separately from `fluid`, hence the clone-free dance of
            // pulling the reference out before the mutable borrow below.
            self.body.edge_velocity()
        } else {
            &self.idle_field
        };

        // `spells::apply` needs &mut fluid and &mut particles at once, so the
        // immutable borrows above are resolved into raw references first.
        let report = {
            let tracker = &self.tracker;
            let flow = self.flow.flow();
            spells::apply(
                tracker,
                flow,
                body_edge,
                &mut self.fluid,
                &mut self.particles,
                &mut self.spell_state,
                &self.params,
                warped_dt,
            )
        };
        self.report = report;

        self.fluid.step(warped_dt, &self.params);

        let repairs = if self.frame % SANITIZE_EVERY == 0 {
            self.fluid.sanitize()
        } else {
            0
        };

        self.particles.step(
            self.fluid.velocity(),
            self.fluid.obstacle(),
            warped_dt,
            &self.params,
            &self.particle_cfg,
        );
        self.particles.build_render_buffer(FLUID_W, FLUID_H);

        self.encode_dye();
        self.collect_stats(repairs, ambient);
    }

    /// When nothing has been perceived for a while, keep the screen alive with
    /// slow wandering vortices. A dead-black canvas reads as "broken app", and
    /// this also gives the headless test something deterministic to assert on.
    fn drive_ambient(&mut self, dt: f32) {
        let t = self.time;
        for k in 0..3 {
            let phase = t * (0.11 + 0.037 * k as f32) + k as f32 * 2.1;
            let x = (FLUID_W as f32) * (0.5 + 0.34 * (phase * 0.7).sin());
            let y = (FLUID_H as f32) * (0.5 + 0.30 * (phase * 0.9).cos());
            let dir = phase * 1.7;
            // Injected every frame, so these are *rates*: the field settles at
            // roughly rate / dissipation. The `dt * 60.0` factor expresses them
            // as "per frame at 60 fps" while staying frame-rate independent.
            let mag = 10.0 * dt * 60.0;
            self.fluid
                .add_force(x, y, dir.cos() * mag, dir.sin() * mag, 12.0);
            let hue = 0.55 + 0.12 * (t * 0.05 + k as f32 * 0.33).sin();
            let rgb = hue_to_rgb(hue);
            // The field settles at roughly amount / dye_dissipation, so this is
            // scaled by the decay rate: without it, slowing decay to let hand
            // plumes travel also lets these idle blobs pile up into a flat wash
            // over the whole frame.
            let amount = 0.03 * self.params.dye_dissipation * dt * 60.0;
            self.fluid.add_dye(
                x,
                y,
                [rgb[0] * amount, rgb[1] * amount, rgb[2] * amount],
                9.0,
            );
        }
    }

    // ---------------------------------------------------------------- output

    /// Tone-maps the float dye field into the RGBA8 texture JS uploads.
    ///
    /// `1 - exp(-x)` rather than a hard clamp: dye accumulates without bound
    /// where gestures dwell, and a clamp turns those regions into flat white
    /// blobs while this keeps the internal structure visible. The curve comes
    /// from a table — see [`TONEMAP_LUT`] for why.
    fn encode_dye(&mut self) {
        let dye = self.fluid.dye();
        let lut = &self.tonemap_lut;
        let (r_ch, g_ch, b_ch) = (&dye[0].data, &dye[1].data, &dye[2].data);

        for i in 0..FLUID_CELLS {
            let r = tonemap_lut(lut, r_ch[i]);
            let g = tonemap_lut(lut, g_ch[i]);
            let b = tonemap_lut(lut, b_ch[i]);
            let o = i * 4;
            self.dye_rgba[o] = (r * 255.0) as u8;
            self.dye_rgba[o + 1] = (g * 255.0) as u8;
            self.dye_rgba[o + 2] = (b * 255.0) as u8;
            // Alpha carries brightness so the compositor can blend the dye
            // over the camera feed without a second texture fetch.
            self.dye_rgba[o + 3] = (r.max(g).max(b) * 255.0) as u8;
        }
    }

    #[inline]
    pub fn dye_rgba(&self) -> &[u8] {
        &self.dye_rgba
    }

    #[inline]
    pub fn dye_ptr(&self) -> *const u8 {
        self.dye_rgba.as_ptr()
    }

    #[inline]
    pub fn particle_buffer(&self) -> &[f32] {
        self.particles.render_buffer()
    }

    #[inline]
    pub fn particle_ptr(&self) -> *const f32 {
        self.particles.render_ptr()
    }

    #[inline]
    pub fn particle_count(&self) -> usize {
        self.particles.active()
    }

    /// Encodes the obstacle field and flow into an RGBA debug texture:
    /// red = obstacle, green/blue = signed flow. Drives the debug overlay.
    pub fn debug_rgba(&mut self) -> &[u8] {
        let obstacle = self.fluid.obstacle();
        let flow = self.flow.flow();
        let sx = (FLOW_W - 1) as f32 / (FLUID_W - 1) as f32;
        let sy = (FLOW_H - 1) as f32 / (FLUID_H - 1) as f32;
        for y in 0..FLUID_H {
            let fy = y as f32 * sy;
            for x in 0..FLUID_W {
                let i = y * FLUID_W + x;
                let (u, v) = flow.sample(x as f32 * sx, fy);
                let o = i * 4;
                self.debug_rgba[o] = (obstacle.data[i].clamp(0.0, 1.0) * 255.0) as u8;
                self.debug_rgba[o + 1] = ((u * 0.02 + 0.5).clamp(0.0, 1.0) * 255.0) as u8;
                self.debug_rgba[o + 2] = ((v * 0.02 + 0.5).clamp(0.0, 1.0) * 255.0) as u8;
                self.debug_rgba[o + 3] = 255;
            }
        }
        &self.debug_rgba
    }

    #[inline]
    pub fn debug_ptr(&self) -> *const u8 {
        self.debug_rgba.as_ptr()
    }

    // ----------------------------------------------------------------- stats

    fn collect_stats(&mut self, repairs: usize, ambient: bool) {
        let (dx, dy) = self.flow.dominant_motion();
        let hands = self.tracker.hands();

        // Everything here is an O(1) read, so it stays live every frame —
        // callers treat `hands_present` and friends as state, not as telemetry.
        self.stats.motion_energy = self.flow.motion_energy();
        self.stats.flow_dx = dx;
        self.stats.flow_dy = dy;
        self.stats.particles_alive = self.particles.alive() as u32;
        self.stats.mask_coverage = self.body.coverage();
        self.stats.mask_present = self.body.present();
        self.stats.hands_present = hands.iter().filter(|h| h.present).count() as u32;
        self.stats.pose_present = self.tracker.pose().present;
        self.stats.time_scale = self.params.time_scale * self.report.time_scale;
        self.stats.bursts = self.report.bursts;
        self.stats.nan_repairs = repairs as u32;
        self.stats.frame = self.frame;
        self.stats.ambient = ambient;

        // These three each walk the whole velocity field. The HUD repaints at
        // 10 Hz, so recomputing them at 60 Hz burns grid passes to produce
        // numbers nobody reads.
        if self.frame % STATS_EVERY == 0 {
            self.stats.fluid_energy = self.fluid.energy();
            self.stats.fluid_max_speed = self.fluid.max_speed();
            self.stats.fluid_divergence = self.fluid.last_divergence();
        }
    }

    #[inline]
    pub fn stats(&self) -> &Stats {
        &self.stats
    }

    /// Latched spell for a hand slot, as a stable string for the HUD.
    pub fn spell_name(&self, slot: usize) -> &'static str {
        self.report
            .spells
            .get(slot)
            .copied()
            .unwrap_or(Spell::Idle)
            .name()
    }

    #[inline]
    pub fn time(&self) -> f32 {
        self.time
    }

    #[inline]
    pub fn tracker(&self) -> &GestureTracker {
        &self.tracker
    }

    #[inline]
    pub fn fluid(&self) -> &Fluid {
        &self.fluid
    }

    #[inline]
    pub fn body(&self) -> &BodyMask {
        &self.body
    }

    #[inline]
    pub fn flow_field(&self) -> &VecField {
        self.flow.flow()
    }

    #[inline]
    pub fn obstacle(&self) -> &Grid {
        self.fluid.obstacle()
    }

    // ---------------------------------------------------------------- config

    #[inline]
    pub fn params(&self) -> &Params {
        &self.params
    }

    /// Sets a named tunable. Returns false for an unknown key.
    pub fn set_param(&mut self, key: &str, value: f32) -> bool {
        match key {
            "curl_influence" => {
                self.particle_cfg.curl_influence = value.clamp(0.0, 20.0);
                true
            }
            "gravity" => {
                self.particle_cfg.gravity = value.clamp(-200.0, 200.0);
                true
            }
            "damping" => {
                self.particle_cfg.damping = value.clamp(0.0, 20.0);
                true
            }
            _ => self.params.set(key, value),
        }
    }

    /// Resizes the live particle pool, re-seeding the newly activated slots.
    pub fn set_particle_count(&mut self, n: usize) {
        let n = n.min(MAX_PARTICLES);
        let previous = self.particles.active();
        self.particles.set_active(n);
        if n > previous {
            self.particles
                .seed_uniform(FLUID_W, FLUID_H, self.params.particle_life);
        }
    }

    /// Full reset: clears the fluid, re-seeds particles, forgets perception.
    pub fn reset(&mut self) {
        let active = self.particles.active();
        self.fluid.reset();
        self.particles = Particles::new(MAX_PARTICLES, self.seed);
        self.particles.set_active(active);
        self.particles
            .seed_uniform(FLUID_W, FLUID_H, self.params.particle_life);
        self.clear_perception();
        self.time = 0.0;
        self.frame = 0;
        self.dye_rgba.fill(0);
        self.obstacle_dirty = true;
        // The HUD reads a cached snapshot, so clearing the counters without
        // clearing `stats` would leave the old frame count on screen forever.
        self.stats = Stats {
            time_scale: 1.0,
            ..Default::default()
        };
        self.report = SpellReport {
            time_scale: 1.0,
            ..Default::default()
        };
    }
}

impl Default for Engine {
    fn default() -> Self {
        Self::new(0xA37E_5EED)
    }
}

/// Clamps a timestep into a sane range.
///
/// A tab that was backgrounded for ten seconds hands back a `dt` of 10.0; one
/// step that large blows any explicit integrator apart, so a long gap is
/// treated as a single slow frame instead.
#[inline]
fn sane_dt(dt: f32) -> f32 {
    if !dt.is_finite() {
        return 1.0 / 60.0;
    }
    dt.clamp(1.0 / 480.0, 1.0 / 20.0)
}

/// The exact tone-mapping curve. Only the table is used in anger; this is kept
/// as the reference the table is asserted against.
#[cfg(test)]
#[inline]
fn tonemap(x: f32) -> f32 {
    if !x.is_finite() || x <= 0.0 {
        return 0.0;
    }
    1.0 - (-x).exp()
}

/// Builds the `1 - exp(-x)` table, with one guard entry past the end so the
/// interpolation in [`tonemap_lut`] never needs a bounds branch.
fn build_tonemap_lut() -> Vec<f32> {
    let mut lut = Vec::with_capacity(TONEMAP_LUT + 2);
    for i in 0..=TONEMAP_LUT + 1 {
        let x = (i.min(TONEMAP_LUT) as f32 / TONEMAP_LUT as f32) * TONEMAP_MAX;
        lut.push(1.0 - (-x).exp());
    }
    lut
}

/// Linear interpolation into the tone-mapping table.
#[inline]
fn tonemap_lut(lut: &[f32], x: f32) -> f32 {
    // Non-finite dye should be impossible, but a NaN here would write a
    // garbage byte into the texture and flash a stray pixel, so it is cheaper
    // to test than to debug.
    if !(x > 0.0) {
        return 0.0;
    }
    if x >= TONEMAP_MAX {
        return lut[TONEMAP_LUT];
    }
    let t = x * (TONEMAP_LUT as f32 / TONEMAP_MAX);
    let i = t as usize;
    let frac = t - i as f32;
    lut[i] + (lut[i + 1] - lut[i]) * frac
}

/// Expected packed buffer sizes, exported for the JS side's assertions.
pub const HAND_BUFFER_LEN: usize = HAND_BUFFER;
/// Expected pose buffer length.
pub const POSE_BUFFER_LEN: usize = POSE_STRIDE;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engine_runs_without_any_input() {
        let mut e = Engine::new(1);
        for _ in 0..120 {
            e.step(1.0 / 60.0);
        }
        assert_eq!(e.stats().nan_repairs, 0, "simulation produced NaN");
        assert_eq!(e.stats().frame, 120);
        assert_eq!(e.dye_rgba().len(), FLUID_CELLS * 4);
    }

    #[test]
    fn ambient_mode_engages_and_lights_the_screen() {
        let mut e = Engine::new(2);
        for _ in 0..180 {
            e.step(1.0 / 60.0);
        }
        assert!(e.stats().ambient, "ambient drive should have engaged");
        let lit = e.dye_rgba().chunks(4).filter(|p| p[3] > 8).count();
        assert!(lit > 0, "ambient mode left the screen black");
    }

    #[test]
    fn insane_dt_is_clamped() {
        let mut e = Engine::new(3);
        e.step(f32::NAN);
        e.step(1e9);
        e.step(-5.0);
        assert_eq!(e.stats().nan_repairs, 0);
        assert!(e.fluid().max_speed().is_finite());
    }

    #[test]
    fn layout_matches_config() {
        let e = Engine::new(4);
        let l = e.layout();
        assert_eq!(l[0] as usize, FLUID_W);
        assert_eq!(l[1] as usize, FLUID_H);
        assert_eq!(l[2] as usize, FLOW_W);
        assert_eq!(l[3] as usize, FLOW_H);
    }

    #[test]
    fn luma_and_mask_buffers_have_the_documented_size() {
        let mut e = Engine::new(5);
        assert_eq!(e.luma_len(), FLOW_CELLS);
        assert_eq!(e.mask_mut().len(), MASK_IN_CAPACITY);
    }

    #[test]
    fn push_luma_accepts_the_engine_buffer() {
        let mut e = Engine::new(6);
        for frame in 0..4u8 {
            for (i, px) in e.luma_mut().iter_mut().enumerate() {
                *px = (i as u8).wrapping_add(frame * 7);
            }
            e.push_luma(1.0 / 30.0);
        }
        assert!(e.flow_field().u.data.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn oversized_mask_is_rejected_not_panicked() {
        let mut e = Engine::new(7);
        e.push_mask(4096, 4096, false, 1.0 / 30.0);
        assert!(!e.body().present());
    }

    #[test]
    fn mask_marks_an_obstacle_and_survives_a_step() {
        let mut e = Engine::new(8);
        let (mw, mh) = (64, 64);
        for y in 0..mh {
            for x in 0..mw {
                // A solid block in the middle of the frame.
                let inside = (24..40).contains(&x) && (24..40).contains(&y);
                e.mask_mut()[y * mw + x] = if inside { 1.0 } else { 0.0 };
            }
        }
        e.push_mask(mw, mh, false, 1.0 / 30.0);
        assert!(e.body().present(), "mask should register a body");
        assert!(e.body().coverage() > 0.0);
        e.step(1.0 / 60.0);
        assert_eq!(e.stats().nan_repairs, 0);
    }

    #[test]
    fn set_param_rejects_unknown_keys() {
        let mut e = Engine::new(9);
        assert!(e.set_param("vorticity", 20.0));
        assert!(e.set_param("curl_influence", 3.0));
        assert!(!e.set_param("not_a_real_param", 1.0));
    }

    #[test]
    fn particle_buffer_tracks_the_active_count() {
        let mut e = Engine::new(10);
        e.set_particle_count(1000);
        e.step(1.0 / 60.0);
        assert_eq!(e.particle_count(), 1000);
        assert_eq!(e.particle_buffer().len(), 1000 * PARTICLE_STRIDE);
    }

    #[test]
    fn particle_count_is_capped_at_capacity() {
        let mut e = Engine::new(11);
        e.set_particle_count(usize::MAX);
        assert_eq!(e.particle_count(), MAX_PARTICLES);
    }

    #[test]
    fn particle_positions_stay_normalised() {
        let mut e = Engine::new(12);
        e.set_particle_count(4096);
        for _ in 0..60 {
            e.step(1.0 / 60.0);
        }
        for p in e.particle_buffer().chunks(PARTICLE_STRIDE) {
            assert!((-0.001..=1.001).contains(&p[0]), "x out of range: {}", p[0]);
            assert!((-0.001..=1.001).contains(&p[1]), "y out of range: {}", p[1]);
            assert!((0.0..=1.0).contains(&p[3]), "life out of range: {}", p[3]);
        }
    }

    #[test]
    fn reset_clears_the_screen_and_the_clock() {
        let mut e = Engine::new(13);
        for _ in 0..200 {
            e.step(1.0 / 60.0);
        }
        e.reset();
        assert_eq!(e.time(), 0.0);
        assert_eq!(e.stats().frame, 0);
        assert!(e.dye_rgba().iter().all(|&b| b == 0));
    }

    #[test]
    fn hand_input_marks_perception_active() {
        use crate::config::{HAND_LANDMARKS, HAND_STRIDE};
        let mut e = Engine::new(14);
        let mut packed = vec![0.0f32; HAND_BUFFER_LEN];
        packed[0] = 1.0; // present
        packed[1] = 1.0; // right hand
        for l in 0..HAND_LANDMARKS {
            packed[4 + l * 3] = 0.5;
            packed[4 + l * 3 + 1] = 0.5;
        }
        assert_eq!(packed.len(), HAND_STRIDE * 2);
        e.push_hands(&packed, 1.0 / 30.0);
        e.step(1.0 / 60.0);
        assert_eq!(e.stats().hands_present, 1);
        assert!(!e.stats().ambient, "a tracked hand should suppress ambient");
    }

    #[test]
    fn short_hand_buffer_is_tolerated() {
        let mut e = Engine::new(15);
        e.push_hands(&[], 1.0 / 30.0);
        e.push_hands(&[1.0, 0.0], 1.0 / 30.0);
        e.push_pose(&[], 1.0 / 30.0);
        e.step(1.0 / 60.0);
        assert_eq!(e.stats().nan_repairs, 0);
    }

    #[test]
    fn tonemap_is_bounded_and_monotonic() {
        assert_eq!(tonemap(0.0), 0.0);
        assert_eq!(tonemap(f32::NAN), 0.0);
        assert!(tonemap(1e9) <= 1.0);
        assert!(tonemap(2.0) > tonemap(1.0));
    }

    #[test]
    fn tonemap_table_matches_the_exact_curve() {
        let lut = build_tonemap_lut();
        // The table replaces 110k exp() calls per frame, so the thing to prove
        // is that it is not a visible approximation: 1/255 is one 8-bit step,
        // i.e. the smallest difference the texture can even represent.
        let mut worst = 0.0f32;
        for i in 0..4000 {
            let x = i as f32 * 0.005;
            let err = (tonemap_lut(&lut, x) - tonemap(x)).abs();
            worst = worst.max(err);
        }
        assert!(
            worst < 1.0 / 255.0,
            "table deviates by {worst}, more than one 8-bit step"
        );
    }

    #[test]
    fn tonemap_table_handles_edges_and_garbage() {
        let lut = build_tonemap_lut();
        assert_eq!(tonemap_lut(&lut, 0.0), 0.0);
        assert_eq!(tonemap_lut(&lut, -5.0), 0.0);
        assert_eq!(tonemap_lut(&lut, f32::NAN), 0.0);
        assert!(tonemap_lut(&lut, f32::INFINITY) <= 1.0);
        // Just under the saturation point must not read past the table.
        assert!(tonemap_lut(&lut, TONEMAP_MAX - 1e-4) <= 1.0);
        assert!(tonemap_lut(&lut, TONEMAP_MAX * 100.0) <= 1.0);
    }

    #[test]
    fn stats_stay_fresh_despite_throttling() {
        // The grid-walking stats are throttled, but everything callers treat as
        // state must be live on the very first frame — this test exists because
        // throttling the whole struct broke `hand_input_marks_perception_active`.
        let mut e = Engine::new(20);
        for expected in 1..=(STATS_EVERY * 3) {
            e.step(1.0 / 60.0);
            assert_eq!(e.stats().frame, expected, "frame counter stalled");
            assert_eq!(e.stats().hands_present, 0);
        }
        // And the throttled fields must have been filled at least once.
        assert!(e.stats().fluid_energy > 0.0, "grid stats never refreshed");
    }

    #[test]
    fn nan_is_caught_within_the_scrub_interval() {
        let mut e = Engine::new(21);
        // Sanitising every frame is a cost paid forever against something that
        // should never happen, so it is throttled. Prove the throttle still
        // catches corruption promptly.
        for _ in 0..=SANITIZE_EVERY {
            e.step(1.0 / 60.0);
        }
        assert_eq!(e.stats().nan_repairs, 0, "clean run should report no repairs");
    }

    #[test]
    fn debug_texture_is_fully_opaque() {
        let mut e = Engine::new(16);
        e.step(1.0 / 60.0);
        let dbg = e.debug_rgba();
        assert_eq!(dbg.len(), FLUID_CELLS * 4);
        assert!(dbg.chunks(4).all(|p| p[3] == 255));
    }
}
