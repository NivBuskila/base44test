/**
 * The base layer: treated camera behind an upscaled dye field, written as
 * linear radiance into the HDR scene target.
 *
 * Two things here are load-bearing for how the project looks:
 *
 * 1. The dye field is 256x144 and the canvas can be 4K. A bilinear stretch of
 *    that is unmistakably a stretched 256-wide image — you can see the lattice
 *    as soft diamonds. This samples it with a cubic B-spline (four bilinear
 *    taps) through a sub-cell domain warp, then modulates the result with
 *    multiplicative noise. Same data, but it reads as turbulence rather than as
 *    an upscale.
 * 2. The engine packed the field through `1 - exp(-x)` to fit it into 8 bits.
 *    Inverting that curve here recovers the pre-clip dynamic range, so dense
 *    cores come back as values well above 1.0 and actually bloom, instead of
 *    flattening into pastel blobs.
 */

import { LUMA } from './common';

export const SCENE_FRAG = /* glsl */ `
precision highp float;

in vec2 v_uv;
out vec4 o_color;

uniform sampler2D u_dye;
uniform sampler2D u_video;
uniform sampler2D u_noise;
/** Dye grid dimensions in cells, and their reciprocal. */
uniform vec2 u_dyeSize;
uniform vec2 u_dyeTexel;
/** Cover-fit crop for the camera, applied around the centre of the frame. */
uniform vec2 u_camScale;
uniform vec2 u_videoTexel;
uniform vec3 u_camTint;
uniform vec3 u_camEdge;
/** How hard the camera's toe is crushed: 1 buries the room, 0 leaves it linear. */
uniform float u_camToe;
/** 1 shows the feed as-is (full colour, no soft focus or rim); 0 treats it. */
uniform float u_camRaw;
uniform float u_dyeAmount;
uniform float u_bgAmount;
uniform float u_hasVideo;
uniform float u_intensity;
uniform float u_time;
${LUMA}

/**
 * Cubic B-spline resample in four bilinear taps (Sigg & Hadwiger).
 *
 * A 16-tap Catmull-Rom is sharper, but it rings around the dye's hard bright
 * cores and the overshoot shows as a dark halo. The B-spline is C2 continuous,
 * which is exactly what kills the visible lattice; the slight extra softness is
 * paid back by the warp and the grain below.
 */
vec4 bspline(sampler2D tex, vec2 uv, vec2 size, vec2 texel) {
  vec2 coord = uv * size;
  vec2 centre = floor(coord - 0.5) + 0.5;
  vec2 f = coord - centre;
  vec2 f2 = f * f;
  vec2 f3 = f2 * f;
  vec2 w0 = (1.0 / 6.0) * (-f3 + 3.0 * f2 - 3.0 * f + 1.0);
  vec2 w1 = (1.0 / 6.0) * (3.0 * f3 - 6.0 * f2 + 4.0);
  vec2 w2 = (1.0 / 6.0) * (-3.0 * f3 + 3.0 * f2 + 3.0 * f + 1.0);
  vec2 w3 = (1.0 / 6.0) * f3;
  // Both partial sums stay in [1/6, 5/6] over f in [0, 1], so the divides
  // below can never blow up.
  vec2 s0 = w0 + w1;
  vec2 s1 = w2 + w3;
  vec2 t0 = (centre - 1.0 + w1 / s0) * texel;
  vec2 t1 = (centre + 1.0 + w3 / s1) * texel;
  vec4 a = texture(tex, vec2(t0.x, t0.y));
  vec4 b = texture(tex, vec2(t1.x, t0.y));
  vec4 c = texture(tex, vec2(t0.x, t1.y));
  vec4 d = texture(tex, vec2(t1.x, t1.y));
  return mix(mix(a, b, s1.x), mix(c, d, s1.x), s1.y);
}

/** Noise lookup expressed in dye cells: one tile texel per \`cells\` cells. */
vec4 cellNoise(vec2 cell, float cells) {
  return texture(u_noise, cell / (cells * NOISE_SIZE));
}

vec3 dyeRadiance(vec2 uv) {
  vec2 cell = uv * u_dyeSize;

  // Warp by a fraction of a cell before sampling. The field drifts slowly so
  // the structure breathes instead of sitting still like a texture overlay.
  vec2 warp = cellNoise(cell + vec2(u_time * 1.7, -u_time * 1.3), 3.0).rg - 0.5;
  vec2 warped = uv + warp * u_dyeTexel * 2.4;

  vec4 s = bspline(u_dye, warped, u_dyeSize, u_dyeTexel);

  // Undo the engine's 1 - exp(-x) packing. The 0.996 keeps the log finite at
  // 255 while still reaching ~5.5 radiance, which is deep into bloom territory.
  vec3 lin = -log(max(vec3(1.0) - s.rgb * 0.996, vec3(1.0e-3)));

  // Structure is applied multiplicatively so empty cells stay empty: an
  // additive term would fog the whole frame, and most of this frame is empty.
  //
  // Two scales, and the choice of channel matters. The smooth fBm channels
  // sampled near their own texel rate give wisps a few pixels across; the
  // white-noise channels sampled through a filter at that rate instead clump
  // into 2x2 blocks, which is unmistakably digital. Per-pixel grain therefore
  // comes from an unfiltered fetch at exactly one texel per device pixel,
  // where there is no filter to clump it.
  vec2 wcell = warped * u_dyeSize;
  float fine = cellNoise(wcell + vec2(0.0, u_time * 0.9), 0.55).g;
  float coarse = cellNoise(wcell - vec2(u_time * 0.5, 0.0), 26.0).r;
  ivec2 grainAt = ivec2(gl_FragCoord.xy) + ivec2(int(u_time * 24.0) * 53, int(u_time * 24.0) * 19);
  float film = texelFetch(u_noise, grainAt & ivec2(NOISE_MASK), 0).b;
  lin *= (0.84 + 0.32 * fine * (0.4 + 1.2 * coarse)) * (0.92 + 0.16 * film);

  // Over-saturate on the way in: the filmic curve in the final pass pulls
  // everything toward white, and a field graded to look right here comes out
  // washed there.
  lin = max(mix(vec3(luma(lin)), lin, 1.24), vec3(0.0));

  // Alpha carries the engine's own brightness. The tenth power keeps this
  // white-hot core off everything except cells that are genuinely saturated —
  // at a lower exponent it whitens the whole ridge of every arm and the field
  // loses its colour.
  lin += vec3(0.55, 0.78, 1.0) * pow(s.a, 10.0) * 0.9;

  return lin * (0.42 + 0.34 * u_intensity);
}

/**
 * Camera, treated rather than shown.
 *
 * Five taps: a centre plus a two-texel cross. The cross doubles as a cheap
 * soft focus and as the edge detector for the silhouette rim, so the subject
 * can be outlined without a second pass.
 *
 * The toe is what makes the layer work in the aether view. A room lit well
 * enough for hand tracking sits around 0.4-0.6 luma across the whole frame,
 * and passing that through linearly floods the screen with a flat wash the
 * fluid then has to compete with. The near-cubic toe drops that wash to
 * nothing and keeps only the subject's highlights and the silhouette rim; the
 * camera view mixes most of it back out, because there the feed is the point.
 */
vec3 treatedCamera(vec2 iuv) {
  vec2 o = u_videoTexel * 2.0;
  float c = luma(texture(u_video, iuv).rgb);
  float xl = luma(texture(u_video, iuv - vec2(o.x, 0.0)).rgb);
  float xr = luma(texture(u_video, iuv + vec2(o.x, 0.0)).rgb);
  float yd = luma(texture(u_video, iuv - vec2(0.0, o.y)).rgb);
  float yu = luma(texture(u_video, iuv + vec2(0.0, o.y)).rgb);
  float soft = (2.0 * c + xl + xr + yd + yu) * (1.0 / 6.0);
  float edge = min(1.0, (abs(xr - xl) + abs(yu - yd)) * 1.7);
  float body = mix(soft, soft * soft * (0.40 + 0.60 * soft), u_camToe);
  // The feed is sRGB-encoded, but everything past this point is linear
  // radiance that the final pass re-encodes. Left as-is the camera gets the
  // transfer twice and turns into a flat, milky wash. The toe is tuned on the
  // encoded value (it is a perceptual crush), so the decode is applied to the
  // part of the feed the toe leaves alone: toe 1 is unchanged, toe 0 is a
  // properly linear feed.
  body = mix(pow(body, 2.2), body, u_camToe);
  return u_camTint * body + u_camEdge * edge;
}

void main() {
  // Row 0 of the dye field is the top of the screen, but GL samples with y up.
  // Every source is read with v flipped here rather than flipping the uploads,
  // which would cost a copy per frame.
  vec2 duv = vec2(v_uv.x, 1.0 - v_uv.y);
  vec3 radiance = vec3(0.0);

  if (u_bgAmount > 0.0) {
    // These are linear radiances, and the sRGB transfer at the end of the
    // chain lifts them hard: 0.004 linear is already a visible navy. The whole
    // background budget is therefore in the third decimal place.
    float r = length((v_uv - vec2(0.5, 0.54)) * vec2(1.0, 0.82));
    vec3 base = mix(vec3(0.00120, 0.00185, 0.00460), vec3(0.00016, 0.00024, 0.00070),
                    smoothstep(0.06, 0.78, r));
    // A slow nebula keeps an idle frame from reading as a dead flat gradient.
    float neb = texture(u_noise, duv * vec2(0.22, 0.13) + vec2(u_time * 0.004, -u_time * 0.003)).r;
    base += vec3(0.00035, 0.00110, 0.00330) * neb * (0.3 + 1.2 * u_intensity);
    radiance += base * u_bgAmount;
  }

  vec3 dye = u_dyeAmount > 0.0 ? dyeRadiance(duv) * u_dyeAmount : vec3(0.0);

  if (u_hasVideo > 0.5) {
    // Mirrored horizontally: the engine's coordinate convention is the flipped
    // view the user sees, so a raw feed would make every gesture land on the
    // wrong side of the screen.
    vec2 iuv = (vec2(1.0 - v_uv.x, 1.0 - v_uv.y) - 0.5) * u_camScale + 0.5;
    if (u_camRaw > 0.5) {
      // The plain feed: decode the sRGB frame to linear and let the composite
      // re-encode it. No luma collapse, no soft focus, no rim.
      vec3 cam = u_camTint * pow(texture(u_video, iuv).rgb, vec3(2.2));
      // Dense dye occludes the feed like smoke instead of only adding to it:
      // purely additive fluid vanishes against a bright room.
      radiance += cam * exp(-luma(dye) * 2.5);
    } else {
      radiance += treatedCamera(iuv);
    }
  }

  radiance += dye;

#if HDR_FLOAT
  o_color = vec4(radiance, 1.0);
#else
  // On an 8-bit intermediate the dark gradients that make up most of this
  // image band before the final pass ever gets to dither them, so dither the
  // scene write too.
  float d = texelFetch(u_noise, ivec2(gl_FragCoord.xy) & ivec2(NOISE_MASK), 0).a - 0.5;
  o_color = vec4(EMIT(radiance) + d * (1.0 / 255.0), 1.0);
#endif
}`;
