/**
 * WebGL2 compositor.
 *
 * One frame, in order:
 *
 * 1. **scene** — treated camera behind a cubic-resampled, domain-warped dye
 *    field, written as linear radiance into an HDR target.
 * 2. **particles** — one `gl.POINTS` draw from a single streamed buffer,
 *    additive into the same target.
 * 3. **overlay** — the hand skeleton, additive, so the bloom picks it up.
 * 4. **bloom** — a 13-tap downsample chain and a 9-tap tent fold-back, five
 *    small draws, driven by `frame.intensity`.
 * 5. **composite** — chromatic aberration, ACES tone map, split tone, vignette,
 *    sRGB transfer and a triangular dither, straight to the default
 *    framebuffer.
 *
 * Invariants the rest of the app depends on:
 * - Nothing is allocated inside `render` on the steady-state path. Textures and
 *   framebuffers are (re)allocated only from `resize`, from the constructor, or
 *   when the camera's resolution changes.
 * - The default framebuffer is bound and fully written when `render` returns,
 *   so `sampleLuminance` and Playwright screenshots see a finished frame.
 * - Every intermediate target survives on either a float or an 8-bit format;
 *   the shaders are compiled against whichever one the context actually
 *   supports.
 */

import { FLUID_H, FLUID_W, MAX_PARTICLES, PARTICLE_STRIDE } from '../constants';
import type { RenderFrame, ViewMode } from '../types';
import { Program, RenderTarget, RendererError, createTexture, isSoftwareRasteriser, probeHdr } from './gl';
import { OVERLAY_CAPACITY, OVERLAY_STRIDE, buildHandMesh } from './handmesh';
import { NOISE_SIZE, buildNoiseTile } from './noise';
import { BLOOM_DOWN_FRAG, BLOOM_UP_FRAG } from './shaders/bloom';
import { FULLSCREEN_VERT, buildShader } from './shaders/common';
import type { ShaderEnv } from './shaders/common';
import { BLIT_FRAG, COMPOSITE_FRAG } from './shaders/composite';
import { OVERLAY_FRAG, OVERLAY_VERT } from './shaders/overlay';
import { PARTICLE_FRAG, PARTICLE_VERT } from './shaders/particles';
import { SCENE_FRAG } from './shaders/scene';

export { RendererError } from './gl';

/**
 * `sampleLuminance` reduces the frame to an NxN block read from across it.
 * A single pixel is not enough: the fluid is sparse, so the centre pixel is
 * legitimately black much of the time and a one-pixel probe cannot tell that
 * apart from a dead render path.
 */
const LUMA_SAMPLE_GRID = 16;

/**
 * Device pixel ratio ceiling. Beyond 2 the fill cost buys nothing visible for a
 * field this soft, and it halves the frame rate on phones.
 */
const MAX_DPR = 2;

/** Deepest bloom mip. Five halvings is a glow radius of ~1/16 of the frame. */
const BLOOM_LEVELS = 5;

/**
 * Ceiling on the scene and bloom pixel count, per frame.
 *
 * The chain is fill-bound: roughly twenty texture fetches per output pixel.
 * That is nothing at 1080p on a GPU and ruinous at 5K on a retina panel with
 * the DPR cap doubling it again, so the scene and the bloom are drawn at
 * whatever fraction keeps them under this budget and the composite scales them
 * back up. The composite itself always runs at full canvas resolution, so the
 * grade, the vignette and the dither stay per-pixel and the result does not
 * read as an upscale.
 *
 * 5.2 Mpx leaves 1440p at DPR 2 untouched and pulls 5K retina back to ~0.6.
 * A CPU rasteriser gets a much tighter budget — but one still above 1280x720,
 * because that is the headless test resolution and the screenshots taken there
 * are a deliverable, not just a smoke check.
 */
const SCENE_PIXEL_BUDGET = 5_200_000;
const SOFTWARE_PIXEL_BUDGET = 1_600_000;

/** Floor on the internal scale; below this the softening is obvious. */
const MIN_SCENE_SCALE = 0.4;

/** Tent filter radius for the bloom fold-back, in source texels. */
const BLOOM_TENT = 1.1;

/**
 * Particle buffer sizes are rounded up to this many particles.
 *
 * The buffer is re-specified every frame to orphan it, and a size that changed
 * every frame would make the driver allocate a differently sized block each
 * time. Bucketing keeps the allocation stable across frames while still
 * tracking a pool the user shrank from 220k to 2k.
 */
const PARTICLE_BUCKET = 32768;

/** Per-view-mode look. Everything that differs between modes lives here. */
interface ModeStyle {
  /** Background gradient and nebula. */
  bg: number;
  /** Dye layer gain. */
  dye: number;
  /** Camera body tint; also decides whether the camera layer runs at all. */
  camTint: readonly [number, number, number];
  /** Camera silhouette rim colour. */
  camEdge: readonly [number, number, number];
  /** Camera toe strength; 1 buries the room, 0 leaves the feed linear. */
  camToe: number;
  particles: number;
  /** Hand skeleton overlay alpha. */
  overlay: number;
  bloom: number;
  /** Radiance above which a highlight blooms. */
  threshold: number;
  exposure: number;
  aberration: number;
  vignette: number;
}

const STYLES: Record<ViewMode, ModeStyle> = {
  // The full composite: the camera is crushed to a cold suggestion of a room
  // and the fluid owns the frame.
  aether: {
    bg: 1,
    dye: 1,
    // Linear radiance, and the sRGB transfer at the end lifts it hard: these
    // put a brightly lit subject at roughly 12% display grey and the
    // silhouette rim at 35%, which is as much room as the fluid can share.
    camTint: [0.0065, 0.0105, 0.0215],
    camEdge: [0.018, 0.048, 0.088],
    camToe: 1,
    particles: 1,
    overlay: 0.35,
    bloom: 0.95,
    threshold: 0.62,
    exposure: 1.0,
    aberration: 0.55,
    vignette: 0.52,
  },
  // Camera-forward: the feed is still treated, just not crushed, and the fluid
  // becomes the faint overlay instead of the subject.
  camera: {
    bg: 0.45,
    dye: 0.34,
    // Slightly warm to cancel the composite's cool split tone on a grey room.
    camTint: [1.02, 1.0, 0.98],
    camEdge: [0.04, 0.09, 0.16],
    camToe: 0.2,
    particles: 0.4,
    overlay: 1.0,
    // The bloom knee is 0.7x the threshold, so at 0.9 anything above ~0.27
    // radiance glowed and a lit wall hazed the whole frame. Only true
    // highlights should bloom over a live feed.
    bloom: 0.15,
    threshold: 1.25,
    exposure: 0.9,
    aberration: 0.3,
    vignette: 0.42,
  },
  // Particles on black, with the bloom pushed: this is the mode where the
  // point cloud has to carry the whole image on its own.
  particles: {
    bg: 0,
    dye: 0,
    camTint: [0, 0, 0],
    camEdge: [0, 0, 0],
    camToe: 1,
    particles: 1.3,
    overlay: 0,
    bloom: 1.25,
    threshold: 0.42,
    exposure: 1.05,
    aberration: 0.6,
    vignette: 0.34,
  },
  // Unused: the debug view bypasses the whole chain. Present so the table is
  // total over ViewMode and a new mode cannot silently fall through.
  debug: {
    bg: 0,
    dye: 0,
    camTint: [0, 0, 0],
    camEdge: [0, 0, 0],
    camToe: 1,
    particles: 0,
    overlay: 0,
    bloom: 0,
    threshold: 1,
    exposure: 1,
    aberration: 0,
    vignette: 0,
  },
};

/** Everything that dies with the GL context and is rebuilt on restore. */
interface Resources {
  scene: RenderTarget;
  bloom: RenderTarget[];
  dyeTex: WebGLTexture;
  debugTex: WebGLTexture;
  videoTex: WebGLTexture;
  noiseTex: WebGLTexture;
  particleBuffer: WebGLBuffer;
  particleVao: WebGLVertexArrayObject;
  overlayBuffer: WebGLBuffer;
  overlayVao: WebGLVertexArrayObject;
  /** Empty VAO for the fullscreen passes, which fetch no attributes. */
  quadVao: WebGLVertexArrayObject;
  scenePass: Program;
  particlePass: Program;
  downPass: Program;
  upPass: Program;
  compositePass: Program;
  overlayPass: Program;
  blitPass: Program;
}

const clamp = (v: number, lo: number, hi: number): number => (v < lo ? lo : v > hi ? hi : v);

export class Renderer {
  private readonly canvas: HTMLCanvasElement;
  private readonly gl: WebGL2RenderingContext;
  /** Forces the 8-bit intermediate path, to exercise the fallback on demand. */
  private readonly forceSdr: boolean;

  private res: Resources | null = null;
  private video: HTMLVideoElement | null = null;
  private lostContext = false;

  private dpr = 1;
  private bloomLevels = 1;
  private frameIndex = 0;
  /** Scene/bloom resolution as a fraction of the canvas; 1 unless capped. */
  private sceneScale = 1;
  /** Scene pixel ceiling, tightened when the context is a CPU rasteriser. */
  private pixelBudget = SCENE_PIXEL_BUDGET;
  /** `?rscale=` override, which wins over the budget. */
  private readonly pinnedScale: number | null;

  /** Video texture allocation state; `0` means "not allocated yet". */
  private videoTexW = 0;
  private videoTexH = 0;
  /** Last uploaded video timestamp, so a 30 Hz feed is not re-uploaded at 60. */
  private videoTime = -1;

  private readonly overlayScratch = new Float32Array(OVERLAY_CAPACITY * OVERLAY_STRIDE);
  /** One-row scratch for `sampleLuminance`, grown to the canvas width. */
  private sampleRow = new Uint8Array(4);

  private readonly onResize = (): void => this.resize();
  private readonly onLost = (e: Event): void => {
    // Without preventDefault the browser never fires `webglcontextrestored`,
    // and a lost context becomes permanent.
    e.preventDefault();
    this.lostContext = true;
    console.warn('[aether] WebGL context lost; waiting for restore');
  };
  private readonly onRestored = (): void => {
    // Every GL object died with the context, so the wrappers holding them are
    // stale: rebuild rather than reuse, and re-probe the formats because a
    // restored context can be a different (often software) implementation.
    this.releaseResources();
    this.videoTexW = 0;
    this.videoTexH = 0;
    this.videoTime = -1;
    try {
      this.res = this.createResources();
      this.lostContext = false;
      console.info('[aether] WebGL context restored');
    } catch (err) {
      console.error('[aether] could not rebuild the renderer after context restore', err);
    }
  };

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    const gl = canvas.getContext('webgl2', {
      alpha: false,
      antialias: false,
      depth: false,
      stencil: false,
      // Needed so `sampleLuminance` and Playwright screenshots can read the
      // framebuffer after the frame is drawn.
      preserveDrawingBuffer: true,
      powerPreference: 'high-performance',
    });
    if (!gl) throw new RendererError('WebGL2 is unavailable in this browser.');
    this.gl = gl;
    // Two diagnostic overrides, both off in normal use: `?sdr8` forces the
    // 8-bit intermediate path and `?rscale=` pins the internal resolution, so
    // the fallbacks can be looked at without the hardware that triggers them.
    const query = typeof location !== 'undefined' ? new URLSearchParams(location.search) : null;
    this.forceSdr = query?.has('sdr8') ?? false;
    const pinned = Number(query?.get('rscale') ?? NaN);
    this.pinnedScale = Number.isFinite(pinned) && pinned > 0 ? clamp(pinned, 0.25, 1) : null;

    canvas.addEventListener('webglcontextlost', this.onLost);
    canvas.addEventListener('webglcontextrestored', this.onRestored);
    window.addEventListener('resize', this.onResize);

    this.res = this.createResources();
    this.resize();
  }

  // ------------------------------------------------------------- resources

  private createResources(): Resources {
    const gl = this.gl;
    const caps = probeHdr(gl, this.forceSdr);
    const env: ShaderEnv = { float: caps.float, range: caps.range };
    if (!caps.float) {
      console.info('[aether] float render targets unavailable; HDR chain on 8-bit targets');
    }
    // Probed here rather than in the constructor because a restored context is
    // often a software one: a GPU reset commonly falls back to SwiftShader.
    this.pixelBudget = isSoftwareRasteriser(gl) ? SOFTWARE_PIXEL_BUDGET : SCENE_PIXEL_BUDGET;

    const vert = buildShader(FULLSCREEN_VERT, env);
    const program = (v: string, f: string, label: string): Program =>
      new Program(gl, buildShader(v, env), buildShader(f, env), label);

    const scene = new RenderTarget(gl, caps.format);
    const bloom: RenderTarget[] = [];
    for (let i = 0; i < BLOOM_LEVELS; i++) bloom.push(new RenderTarget(gl, caps.format));

    const rgba8 = { internal: gl.RGBA8, format: gl.RGBA, type: gl.UNSIGNED_BYTE };
    const dyeTex = createTexture(gl, FLUID_W, FLUID_H, rgba8, gl.LINEAR);
    // NEAREST for the debug view on purpose: it is meant to show the raw
    // 256x144 cells, and interpolating them would hide exactly what it is for.
    const debugTex = createTexture(gl, FLUID_W, FLUID_H, rgba8, gl.NEAREST);
    const videoTex = createTexture(gl, 2, 2, rgba8, gl.LINEAR);
    const noiseTex = createTexture(gl, NOISE_SIZE, NOISE_SIZE, rgba8, gl.LINEAR, gl.REPEAT);
    gl.bindTexture(gl.TEXTURE_2D, noiseTex);
    gl.texSubImage2D(
      gl.TEXTURE_2D,
      0,
      0,
      0,
      NOISE_SIZE,
      NOISE_SIZE,
      gl.RGBA,
      gl.UNSIGNED_BYTE,
      buildNoiseTile(),
    );

    const particleBuffer = gl.createBuffer();
    const overlayBuffer = gl.createBuffer();
    const particleVao = gl.createVertexArray();
    const overlayVao = gl.createVertexArray();
    const quadVao = gl.createVertexArray();
    if (!particleBuffer || !overlayBuffer || !particleVao || !overlayVao || !quadVao) {
      throw new RendererError('could not allocate the vertex buffers');
    }

    gl.bindVertexArray(particleVao);
    gl.bindBuffer(gl.ARRAY_BUFFER, particleBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, PARTICLE_BUCKET * PARTICLE_STRIDE * 4, gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 4, gl.FLOAT, false, PARTICLE_STRIDE * 4, 0);

    gl.bindVertexArray(overlayVao);
    gl.bindBuffer(gl.ARRAY_BUFFER, overlayBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, this.overlayScratch.byteLength, gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, OVERLAY_STRIDE * 4, 0);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 1, gl.FLOAT, false, OVERLAY_STRIDE * 4, 8);
    gl.bindVertexArray(null);

    return {
      scene,
      bloom,
      dyeTex,
      debugTex,
      videoTex,
      noiseTex,
      particleBuffer,
      particleVao,
      overlayBuffer,
      overlayVao,
      quadVao,
      scenePass: new Program(gl, vert, buildShader(SCENE_FRAG, env), 'scene'),
      particlePass: program(PARTICLE_VERT, PARTICLE_FRAG, 'particles'),
      downPass: new Program(gl, vert, buildShader(BLOOM_DOWN_FRAG, env), 'bloom-down'),
      upPass: new Program(gl, vert, buildShader(BLOOM_UP_FRAG, env), 'bloom-up'),
      compositePass: new Program(gl, vert, buildShader(COMPOSITE_FRAG, env), 'composite'),
      overlayPass: program(OVERLAY_VERT, OVERLAY_FRAG, 'overlay'),
      blitPass: new Program(gl, vert, buildShader(BLIT_FRAG, env), 'blit'),
    };
  }

  private releaseResources(): void {
    const res = this.res;
    if (!res) return;
    const gl = this.gl;
    this.res = null;
    res.scene.dispose();
    for (const level of res.bloom) level.dispose();
    for (const tex of [res.dyeTex, res.debugTex, res.videoTex, res.noiseTex]) gl.deleteTexture(tex);
    gl.deleteBuffer(res.particleBuffer);
    gl.deleteBuffer(res.overlayBuffer);
    gl.deleteVertexArray(res.particleVao);
    gl.deleteVertexArray(res.overlayVao);
    gl.deleteVertexArray(res.quadVao);
    for (const p of [
      res.scenePass,
      res.particlePass,
      res.downPass,
      res.upPass,
      res.compositePass,
      res.overlayPass,
      res.blitPass,
    ]) {
      p.dispose();
    }
  }

  /** Releases every GL object this renderer owns and detaches its listeners. */
  dispose(): void {
    this.canvas.removeEventListener('webglcontextlost', this.onLost);
    this.canvas.removeEventListener('webglcontextrestored', this.onRestored);
    window.removeEventListener('resize', this.onResize);
    this.releaseResources();
    this.video = null;
    this.lostContext = true;
  }

  // ------------------------------------------------------------------ size

  /** Matches the drawing buffer and every target to the CSS size and DPR. */
  resize(): void {
    // Ceiling only, no floor. `devicePixelRatio` drops below 1 whenever the
    // page is zoomed out (0.5 at 50% zoom), and flooring it at 1 there
    // allocates four times the pixels the browser asked for — the opposite of
    // what the user requested — and breaks the drawing-buffer size the app's
    // own DPR test asserts. `|| 1` still covers a 0 or undefined ratio.
    this.dpr = Math.min(MAX_DPR, window.devicePixelRatio || 1);
    const w = Math.max(1, Math.round(this.canvas.clientWidth * this.dpr));
    const h = Math.max(1, Math.round(this.canvas.clientHeight * this.dpr));
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
    }

    const res = this.res;
    if (!res) return;

    const pixels = w * h;
    const budgeted = pixels > this.pixelBudget ? Math.sqrt(this.pixelBudget / pixels) : 1;
    this.sceneScale = clamp(this.pinnedScale ?? budgeted, MIN_SCENE_SCALE, 1);
    const sw = Math.max(1, Math.round(w * this.sceneScale));
    const sh = Math.max(1, Math.round(h * this.sceneScale));
    res.scene.resize(sw, sh);
    // The bloom chain stops halving once a level would be too small for the
    // 13-tap kernel to mean anything; on a phone in portrait that is three
    // levels, on a 4K canvas it is the full five.
    let lw = sw;
    let lh = sh;
    let levels = 0;
    while (levels < BLOOM_LEVELS) {
      const nw = Math.max(1, lw >> 1);
      const nh = Math.max(1, lh >> 1);
      if (nw < 8 || nh < 8) break;
      res.bloom[levels].resize(nw, nh);
      lw = nw;
      lh = nh;
      levels++;
    }
    this.bloomLevels = Math.max(1, levels);
  }

  setVideo(video: HTMLVideoElement | null): void {
    this.video = video;
    this.videoTime = -1;
  }

  get hasVideo(): boolean {
    return this.video !== null;
  }

  // ---------------------------------------------------------------- uploads

  /**
   * True when the video texture holds a usable frame.
   *
   * `readyState < 2` is the case that matters: a camera element exists from the
   * moment `getUserMedia` resolves but has no decoded frame for a while after,
   * and uploading one of those is either a no-op or a throw depending on the
   * browser.
   */
  private uploadVideo(res: Resources, v: HTMLVideoElement | null): boolean {
    if (!v || v.readyState < 2 || v.videoWidth === 0 || v.videoHeight === 0) return false;
    const gl = this.gl;
    gl.activeTexture(gl.TEXTURE0);

    if (this.videoTexW !== v.videoWidth || this.videoTexH !== v.videoHeight) {
      // Only reallocation path for the video texture: the camera's resolution
      // is fixed for the life of a stream, so this runs once.
      gl.bindTexture(gl.TEXTURE_2D, res.videoTex);
      gl.texImage2D(
        gl.TEXTURE_2D,
        0,
        gl.RGBA8,
        v.videoWidth,
        v.videoHeight,
        0,
        gl.RGBA,
        gl.UNSIGNED_BYTE,
        null,
      );
      this.videoTexW = v.videoWidth;
      this.videoTexH = v.videoHeight;
      this.videoTime = -1;
    }

    if (v.currentTime !== this.videoTime) {
      gl.bindTexture(gl.TEXTURE_2D, res.videoTex);
      try {
        gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, gl.RGBA, gl.UNSIGNED_BYTE, v);
      } catch {
        // A frame that is not decodable yet, or a cross-origin stream. Keep
        // whatever was uploaded last rather than dropping the camera layer.
        return this.videoTime >= 0;
      }
      this.videoTime = v.currentTime;
    }
    return true;
  }

  /**
   * Uploads one `FLUID_W * FLUID_H` RGBA8 grid. A short buffer means the view
   * over WASM memory was detached by a reallocation this frame — skip the
   * upload and keep the previous contents rather than throwing out of the
   * render loop.
   */
  private uploadGrid(tex: WebGLTexture, data: Uint8Array): boolean {
    if (data.length < FLUID_W * FLUID_H * 4) return false;
    const gl = this.gl;
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, FLUID_W, FLUID_H, gl.RGBA, gl.UNSIGNED_BYTE, data);
    return true;
  }

  // ----------------------------------------------------------------- frame

  render(frame: RenderFrame): void {
    const gl = this.gl;
    if (this.lostContext || gl.isContextLost()) return;
    const res = this.res;
    if (!res) return;

    this.resize();
    this.frameIndex = (this.frameIndex + 1) % 1024;
    const intensity = Number.isFinite(frame.intensity) ? clamp(frame.intensity, 0, 1) : 0;
    const time = Number.isFinite(frame.time) ? frame.time : 0;

    gl.disable(gl.BLEND);
    gl.disable(gl.DEPTH_TEST);
    gl.bindVertexArray(res.quadVao);

    if (frame.mode === 'debug') {
      this.drawDebug(res, frame);
      return;
    }

    const style = STYLES[frame.mode];
    this.drawScene(res, frame, style, intensity, time);
    this.drawParticles(res, frame, style, intensity);
    this.drawOverlay(res, frame, style);
    this.drawBloom(res, style);
    this.drawComposite(res, style, intensity);
  }

  /** Raw obstacle/flow texture, straight to the screen with no grading. */
  private drawDebug(res: Resources, frame: RenderFrame): void {
    const gl = this.gl;
    const source = frame.debug && this.uploadGrid(res.debugTex, frame.debug) ? res.debugTex : null;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    if (!source) {
      // `frame.debug` is null whenever the engine has nothing to show. Say so
      // with a flat field rather than leaving the last frame on screen.
      gl.clearColor(0.03, 0.035, 0.05, 1);
      gl.clear(gl.COLOR_BUFFER_BIT);
      return;
    }
    res.blitPass.use();
    res.blitPass.tex('u_src', 0, source);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }

  private drawScene(
    res: Resources,
    frame: RenderFrame,
    style: ModeStyle,
    intensity: number,
    time: number,
  ): void {
    const gl = this.gl;
    const p = res.scenePass;

    this.uploadGrid(res.dyeTex, frame.dye);
    // The camera mode is a stronger statement than the camera toggle: the user
    // asked to look at the feed, so `showCamera` only gates the aether view.
    //
    // `camUsed` is not a style choice, it is the identity: `particles` tints
    // both the body and the rim to black, so running the layer there costs a
    // full-frame `texSubImage2D` plus five fetches per pixel to add exactly
    // zero. On a software rasteriser that was a quarter of the frame.
    const camUsed =
      style.camTint[0] + style.camTint[1] + style.camTint[2] > 0 ||
      style.camEdge[0] + style.camEdge[1] + style.camEdge[2] > 0;
    const wantCamera = camUsed && (frame.mode === 'camera' || frame.showCamera);
    const hasVideo = wantCamera && this.uploadVideo(res, frame.video ?? this.video);

    res.scene.bind();
    p.use();
    p.tex('u_dye', 0, res.dyeTex);
    p.tex('u_video', 1, res.videoTex);
    p.tex('u_noise', 2, res.noiseTex);
    p.f2('u_dyeSize', FLUID_W, FLUID_H);
    p.f2('u_dyeTexel', 1 / FLUID_W, 1 / FLUID_H);
    p.f1('u_dyeAmount', style.dye);
    p.f1('u_bgAmount', style.bg);
    p.f1('u_intensity', intensity);
    p.f1('u_time', time);
    p.f1('u_hasVideo', hasVideo ? 1 : 0);
    p.f3('u_camTint', style.camTint[0], style.camTint[1], style.camTint[2]);
    p.f3('u_camEdge', style.camEdge[0], style.camEdge[1], style.camEdge[2]);
    p.f1('u_camToe', style.camToe);
    p.f2('u_videoTexel', 1 / Math.max(1, this.videoTexW), 1 / Math.max(1, this.videoTexH));

    // Cover fit: crop the long axis so the feed fills the canvas at its own
    // aspect ratio. Letterboxing would put black bars inside the simulation,
    // and stretching would make gestures land off their landmarks.
    const canvasAspect = this.canvas.width / Math.max(1, this.canvas.height);
    const videoAspect = this.videoTexW / Math.max(1, this.videoTexH);
    if (videoAspect > canvasAspect) {
      p.f2('u_camScale', canvasAspect / videoAspect, 1);
    } else {
      p.f2('u_camScale', 1, videoAspect / canvasAspect);
    }

    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }

  private drawParticles(
    res: Resources,
    frame: RenderFrame,
    style: ModeStyle,
    intensity: number,
  ): void {
    if (style.particles <= 0) return;
    const gl = this.gl;
    const available = Math.floor(frame.particles.length / PARTICLE_STRIDE);
    const requested = Number.isFinite(frame.particleCount) ? Math.floor(frame.particleCount) : 0;
    const count = Math.min(Math.max(0, requested), available, MAX_PARTICLES);
    if (count === 0) return;

    gl.bindVertexArray(res.particleVao);
    gl.bindBuffer(gl.ARRAY_BUFFER, res.particleBuffer);
    // Orphan, then refill. Re-specifying the whole buffer tells the driver the
    // old contents are dead, so this upload never blocks on the GPU still
    // reading last frame's data — at 220k particles that stall is a dropped
    // frame every frame.
    const bucket = Math.min(
      MAX_PARTICLES,
      Math.ceil(count / PARTICLE_BUCKET) * PARTICLE_BUCKET,
    );
    gl.bufferData(gl.ARRAY_BUFFER, bucket * PARTICLE_STRIDE * 4, gl.DYNAMIC_DRAW);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, frame.particles, 0, count * PARTICLE_STRIDE);

    const p = res.particlePass;
    p.use();
    // Energy normalisation: the pool is user-adjustable from 2k to 220k, and
    // without this the 220k setting is a white screen and the 2k setting is
    // nearly invisible. Total emitted light stays roughly constant instead.
    p.f1('u_gain', style.particles * clamp(90_000 / count, 0.3, 2.0));
    // Point size follows the target, not the canvas: the scene can be drawn
    // below canvas resolution, and a size in canvas pixels would then make the
    // dust swell into blobs when the composite scales it back up.
    const px = this.dpr * this.sceneScale;
    p.f1('u_size', px * (3.1 + (1.35 - 3.1) * clamp(count / 200_000, 0, 1)));
    p.f1('u_intensity', intensity);

    res.scene.bind();
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE);
    gl.drawArrays(gl.POINTS, 0, count);
    gl.disable(gl.BLEND);
  }

  private drawOverlay(res: Resources, frame: RenderFrame, style: ModeStyle): void {
    if (style.overlay <= 0 || !frame.hands) return;
    const mesh = buildHandMesh(frame.hands, this.overlayScratch);
    const vertices = mesh.lineVertices + mesh.pointVertices;
    if (vertices === 0) return;

    const gl = this.gl;
    gl.bindVertexArray(res.overlayVao);
    gl.bindBuffer(gl.ARRAY_BUFFER, res.overlayBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, this.overlayScratch.byteLength, gl.DYNAMIC_DRAW);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, this.overlayScratch, 0, vertices * OVERLAY_STRIDE);

    const p = res.overlayPass;
    p.use();
    p.f1('u_alpha', style.overlay);
    p.f3('u_tint', 0.55, 0.88, 1.0);

    res.scene.bind();
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE);
    if (mesh.lineVertices > 0) {
      p.f1('u_round', 0);
      p.f1('u_size', 1);
      gl.drawArrays(gl.LINES, 0, mesh.lineVertices);
    }
    if (mesh.pointVertices > 0) {
      p.f1('u_round', 1);
      p.f1('u_size', 5 * this.dpr * this.sceneScale);
      gl.drawArrays(gl.POINTS, mesh.lineVertices, mesh.pointVertices);
    }
    gl.disable(gl.BLEND);
  }

  private drawBloom(res: Resources, style: ModeStyle): void {
    const gl = this.gl;
    gl.bindVertexArray(res.quadVao);

    const down = res.downPass;
    down.use();
    down.f1('u_knee', Math.max(0.05, style.threshold * 0.7));
    for (let i = 0; i < this.bloomLevels; i++) {
      const src = i === 0 ? res.scene : res.bloom[i - 1];
      down.tex('u_src', 0, src.texture);
      down.f2('u_texel', 1 / src.width, 1 / src.height);
      // Only the first level thresholds; below that everything in the chain is
      // already bright by construction.
      down.f1('u_threshold', i === 0 ? style.threshold : 0);
      down.f1('u_fromScene', i === 0 ? 1 : 0);
      res.bloom[i].bind();
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    }

    const up = res.upPass;
    up.use();
    up.f1('u_amount', 1);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE);
    for (let i = this.bloomLevels - 1; i > 0; i--) {
      const src = res.bloom[i];
      up.tex('u_src', 0, src.texture);
      up.f2('u_texel', BLOOM_TENT / src.width, BLOOM_TENT / src.height);
      res.bloom[i - 1].bind();
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    }
    gl.disable(gl.BLEND);
  }

  private drawComposite(res: Resources, style: ModeStyle, intensity: number): void {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);

    const p = res.compositePass;
    p.use();
    p.tex('u_scene', 0, res.scene.texture);
    p.tex('u_bloom', 1, res.bloom[0].texture);
    p.tex('u_noise', 2, res.noiseTex);
    // Motion drives the glow, not the exposure: pushing exposure with movement
    // makes the whole frame pump, while pushing bloom makes the bright parts
    // bloom harder, which is what "a burst of movement blazes" should feel like.
    p.f1('u_bloomAmount', style.bloom * (0.72 + 0.75 * intensity));
    p.f1('u_exposure', style.exposure * (0.96 + 0.22 * intensity));
    p.f1('u_aberration', style.aberration);
    p.f1('u_vignette', style.vignette);
    p.f1('u_frame', this.frameIndex);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }

  // ------------------------------------------------------------ inspection

  /**
   * Mean luminance over a grid of samples across the rendered frame, `[0, 1]`.
   *
   * Used by the headless tests to tell "something is being drawn" from "the
   * render path is dead". Reads `LUMA_SAMPLE_GRID^2` pixels spread across the
   * frame rather than a contiguous block, so a bright region anywhere
   * registers. One `readPixels` per sampled row: `n` calls instead of `n^2`.
   */
  sampleLuminance(): number {
    const gl = this.gl;
    if (this.lostContext || gl.isContextLost()) return 0;
    const n = LUMA_SAMPLE_GRID;
    const stepX = Math.max(1, Math.floor(this.canvas.width / n));
    const stepY = Math.max(1, Math.floor(this.canvas.height / n));

    const rowBytes = this.canvas.width * 4;
    if (this.sampleRow.length < rowBytes) this.sampleRow = new Uint8Array(rowBytes);

    // The composite pass leaves the default framebuffer bound, but a caller
    // could read between passes, and readPixels would then sample an
    // intermediate target at the wrong size.
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);

    let total = 0;
    let count = 0;
    for (let row = 0; row < n; row++) {
      const y = Math.min(this.canvas.height - 1, row * stepY);
      gl.readPixels(0, y, this.canvas.width, 1, gl.RGBA, gl.UNSIGNED_BYTE, this.sampleRow);
      for (let i = 0; i < n; i++) {
        const p = Math.min(this.canvas.width - 1, i * stepX) * 4;
        total +=
          0.2126 * this.sampleRow[p] +
          0.7152 * this.sampleRow[p + 1] +
          0.0722 * this.sampleRow[p + 2];
        count++;
      }
    }
    return count === 0 ? 0 : total / count / 255;
  }
}
