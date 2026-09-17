//! Shared dimensions and tunables.
//!
//! These constants are the contract between the Rust engine and the browser:
//! `web/src/constants.ts` mirrors them and `Engine::layout()` re-exports them at
//! runtime so a mismatch surfaces as a loud error instead of a garbled texture.

/// Fluid simulation grid width, in cells.
///
/// 192x108 keeps the 16:9 aspect — so cells are square in screen space and no
/// stage needs an anisotropic correction — and costs 56% of the cells that
/// 256x144 does. Measured, that is most of the difference between a 28 ms and a
/// 16 ms engine step in wasm.
///
/// The visible loss is small because the dye field is never shown at its own
/// resolution: the renderer upscales it with a wide tap pattern and
/// domain-warped noise, and all the high-frequency detail the eye reads comes
/// from the 120k particles, which are resolution-independent. Doubling the dye
/// grid instead would buy sharper *smoke edges* at the cost of the frame rate
/// that makes the whole thing feel alive.
pub const FLUID_W: usize = 192;
/// Fluid simulation grid height, in cells.
pub const FLUID_H: usize = 108;
/// Number of cells in the fluid / dye / obstacle grids.
pub const FLUID_CELLS: usize = FLUID_W * FLUID_H;

/// Optical-flow working resolution. The camera luma plane is downscaled to
/// this before flow is computed; smaller is both faster and less noisy.
pub const FLOW_W: usize = 128;
/// Optical-flow working resolution height.
pub const FLOW_H: usize = 72;
/// Number of samples in the luma input buffer.
pub const FLOW_CELLS: usize = FLOW_W * FLOW_H;

/// Hard ceiling on particles; the render buffer is allocated once at this size.
pub const MAX_PARTICLES: usize = 220_000;
/// Particle count the engine starts at.
pub const DEFAULT_PARTICLES: usize = 120_000;
/// Floats per particle in the render buffer: `x, y, heat, life`.
pub const PARTICLE_STRIDE: usize = 4;

/// MediaPipe hand landmark count.
pub const HAND_LANDMARKS: usize = 21;
/// MediaPipe pose landmark count.
pub const POSE_LANDMARKS: usize = 33;
/// Hands the engine tracks simultaneously.
pub const HANDS: usize = 2;

/// Floats per hand in the packed hand buffer:
/// `present, handedness, gesture_id, gesture_score` then 21 * `(x, y, z)`.
pub const HAND_STRIDE: usize = 4 + HAND_LANDMARKS * 3;
/// Total floats in the packed hand buffer.
pub const HAND_BUFFER: usize = HAND_STRIDE * HANDS;
/// Floats in the packed pose buffer: `present` then 33 * `(x, y, z, visibility)`.
pub const POSE_STRIDE: usize = 1 + POSE_LANDMARKS * 4;

// --- Hand landmark indices (MediaPipe ordering) ---
pub const LM_WRIST: usize = 0;
pub const LM_THUMB_TIP: usize = 4;
pub const LM_INDEX_MCP: usize = 5;
pub const LM_INDEX_TIP: usize = 8;
pub const LM_MIDDLE_MCP: usize = 9;
pub const LM_MIDDLE_TIP: usize = 12;
pub const LM_RING_TIP: usize = 16;
pub const LM_PINKY_MCP: usize = 17;
pub const LM_PINKY_TIP: usize = 20;

// --- Pose landmark indices (MediaPipe ordering) ---
pub const PL_NOSE: usize = 0;
pub const PL_LEFT_SHOULDER: usize = 11;
pub const PL_RIGHT_SHOULDER: usize = 12;
pub const PL_LEFT_WRIST: usize = 15;
pub const PL_RIGHT_WRIST: usize = 16;
pub const PL_LEFT_HIP: usize = 23;
pub const PL_RIGHT_HIP: usize = 24;

/// Runtime-tunable simulation parameters, all settable from the HUD via
/// [`crate::engine::Engine::set_param`].
#[derive(Clone, Copy, Debug)]
pub struct Params {
    /// Velocity field decay per second (1.0 = no decay).
    pub velocity_dissipation: f32,
    /// Dye decay per second.
    pub dye_dissipation: f32,
    /// Jacobi iterations for the pressure projection.
    pub pressure_iters: usize,
    /// Vorticity confinement strength; puts the curl the grid eats back in.
    pub vorticity: f32,
    /// Kinematic viscosity.
    pub viscosity: f32,
    /// How hard hand motion pushes the fluid.
    pub hand_force: f32,
    /// How hard raw optical flow pushes the fluid (the no-ML drive path).
    pub flow_force: f32,
    /// Multiplier on fluid velocity when advecting particles.
    pub particle_drag: f32,
    /// Particle lifetime in seconds.
    pub particle_life: f32,
    /// Particles respawned per second.
    pub spawn_rate: f32,
    /// Global time scale, driven by the two-hand "time warp" gesture.
    pub time_scale: f32,
    /// Strength of the body-silhouette obstacle. 0 disables body collision.
    pub body_push: f32,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            velocity_dissipation: 0.15,
            // Once raised to 1.0 to stop a held gesture pinning a saturated
            // blob at the tone-map ceiling. That cure was worse than the
            // disease: dye died within a hand's width of the palm, so the field
            // read as a glow stuck to the hand rather than smoke crossing the
            // room. Slow enough now for a plume to actually travel.
            dye_dissipation: 0.6,
            pressure_iters: 28,
            vorticity: 14.0,
            viscosity: 0.000_02,
            hand_force: 1.0,
            flow_force: 0.45,
            particle_drag: 1.0,
            particle_life: 4.5,
            spawn_rate: 30_000.0,
            time_scale: 1.0,
            body_push: 1.0,
        }
    }
}

impl Params {
    /// Applies a HUD parameter by name. Returns `false` for unknown keys so the
    /// caller can surface a typo instead of silently ignoring it.
    pub fn set(&mut self, key: &str, value: f32) -> bool {
        match key {
            "velocity_dissipation" => self.velocity_dissipation = value.clamp(0.0, 10.0),
            "dye_dissipation" => self.dye_dissipation = value.clamp(0.0, 10.0),
            "pressure_iters" => self.pressure_iters = (value as usize).clamp(1, 80),
            "vorticity" => self.vorticity = value.clamp(0.0, 60.0),
            "viscosity" => self.viscosity = value.clamp(0.0, 0.01),
            "hand_force" => self.hand_force = value.clamp(0.0, 8.0),
            "flow_force" => self.flow_force = value.clamp(0.0, 8.0),
            "particle_drag" => self.particle_drag = value.clamp(0.0, 4.0),
            "particle_life" => self.particle_life = value.clamp(0.2, 30.0),
            "spawn_rate" => self.spawn_rate = value.clamp(0.0, 400_000.0),
            "time_scale" => self.time_scale = value.clamp(0.05, 4.0),
            "body_push" => self.body_push = value.clamp(0.0, 4.0),
            _ => return false,
        }
        true
    }
}
