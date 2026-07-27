/**
 * Software-rendering detection and frame-cost tracking for the browse canvas
 * (issue #2695).
 *
 * Some production environments run the browser with hardware acceleration
 * permanently disabled, so every canvas paint rasterizes on the CPU. The
 * browse canvas's pan/zoom animations are designed around GPU-cheap
 * operations — full-canvas smoothed blits, overscanned snapshots up to 4× the
 * viewport, shadow blurs — which software rasterization turns into tens of
 * milliseconds per frame, making zooming near-unusable. Rather than forcing
 * such users to turn animations off entirely, the canvas drops into a
 * "low-effects" mode that keeps every animation running but makes each frame
 * cheap: no overscan buffers, nearest-neighbour blits, no shadow blurs, and a
 * capped device-pixel-ratio.
 *
 * Two signals feed that mode, both in this module:
 *
 * - {@link detectSoftwareRenderer} — an upfront probe of the WebGL renderer
 *   string, which names the software rasterizer (SwiftShader, llvmpipe, …)
 *   when acceleration is off. Catches the common case before the first janky
 *   gesture.
 * - {@link FramePerfMonitor} — a latching EMA over measured frame-paint
 *   durations, for environments the probe can't identify (renderer string
 *   masked, WebGL blocked, or a machine that is simply too slow). Latches
 *   low-effects mode after sustained slow frames.
 */

/**
 * The user-facing "Graphics" choice (the ``browse_graphics`` setting).
 * ``auto`` runs the detection described above; ``full`` and ``reduced`` pin
 * the pipeline, so an admin can force the cheap path on a deployment whose
 * clients the probe can't recognize, or force the rich one if detection ever
 * misfires. Mirrors ``BrowseGraphics`` in ``vtsearch/settings_models.py``.
 */
export type BrowseGraphicsMode = 'auto' | 'full' | 'reduced';

/**
 * Resolve whether to paint cheaply, given the user's setting and what the
 * automatic detection currently believes. Pure, so the three-way precedence
 * is testable without a canvas.
 */
export function resolveLowEffects(mode: BrowseGraphicsMode, detectedSlow: boolean): boolean {
  if (mode === 'full') return false;
  if (mode === 'reduced') return true;
  return detectedSlow;
}

/** EMA threshold (ms) above which frame painting is considered too slow to
 * sustain the full-effects animation pipeline: past this the canvas cannot
 * hold even ~40 fps, so the eased zoom/pan reads as a stutter. */
export const SLOW_FRAME_ENTER_MS = 24;

/** Cap (ms) applied to individual samples before they enter the EMA, so one
 * giant outlier (a GC pause, a backgrounded tab waking up) cannot dominate
 * the average and latch low-effects mode on a machine that is actually fine. */
export const SLOW_FRAME_SAMPLE_CAP_MS = 80;

/** Minimum samples before the monitor will judge: the first few frames after
 * load are polluted by cold caches (fonts, style resolution, JIT warm-up). */
export const SLOW_FRAME_MIN_SAMPLES = 6;

/** EMA smoothing factor. Low enough that a single capped outlier landing on a
 * healthy average (a few ms) cannot cross {@link SLOW_FRAME_ENTER_MS} on its
 * own — it takes a *run* of slow frames to latch. */
const EMA_ALPHA = 0.2;

/**
 * Tracks how long the browse canvas's frames take to paint and latches "slow"
 * once the running average shows the machine cannot keep the full-effects
 * animation pipeline smooth.
 *
 * Latch-only by design: software rendering does not switch on or off
 * mid-session, and the degraded pipeline is much cheaper per frame — so an
 * exit threshold would oscillate (slow → degrade → frames get cheap → restore
 * effects → slow again). Once slow, the component stays in low-effects mode
 * for its lifetime.
 *
 * Pure and framework-free: callers feed it durations (typically
 * `performance.now()` deltas around a paint) and read {@link slow}.
 */
export class FramePerfMonitor {
  private ema = 0;
  private samples = 0;
  private latched: boolean;

  /** `initiallySlow` seeds the latch — pass {@link detectSoftwareRenderer}'s
   * verdict so a recognized software rasterizer degrades from the first
   * frame instead of after the first janky gesture. */
  constructor(initiallySlow = false) {
    this.latched = initiallySlow;
  }

  /** Feed one frame-paint duration (ms). */
  record(frameMs: number): void {
    const sample = Math.min(Math.max(0, frameMs), SLOW_FRAME_SAMPLE_CAP_MS);
    this.samples++;
    this.ema = this.samples === 1 ? sample : this.ema + (sample - this.ema) * EMA_ALPHA;
    if (this.samples >= SLOW_FRAME_MIN_SAMPLES && this.ema > SLOW_FRAME_ENTER_MS) {
      this.latched = true;
    }
  }

  /** Whether the canvas should run in low-effects mode. */
  get slow(): boolean {
    return this.latched;
  }
}

/** Renderer strings that name a CPU rasterizer. SwiftShader is Chrome's
 * software fallback (the one a hardware-acceleration-off deployment gets);
 * llvmpipe/softpipe are Mesa's; "Software" covers ANGLE's generic naming. */
const SOFTWARE_RENDERER_RE = /swiftshader|llvmpipe|softpipe|software/i;

/**
 * Best-effort upfront check for a software-rendered browser. Creates a
 * throwaway WebGL context and inspects the renderer string: with hardware
 * acceleration disabled, browsers fall back to a named CPU rasterizer.
 *
 * Deliberately conservative: when WebGL is unavailable or the renderer string
 * is masked, this returns `false` (unknown ≠ software) and leaves the call to
 * the runtime {@link FramePerfMonitor}, which latches from real frame costs.
 */
export function detectSoftwareRenderer(): boolean {
  try {
    const canvas = document.createElement('canvas');
    const gl = (canvas.getContext('webgl') ??
      canvas.getContext('experimental-webgl')) as WebGLRenderingContext | null;
    if (!gl) return false;
    const dbg = gl.getExtension('WEBGL_debug_renderer_info');
    const renderer = String(
      dbg ? gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER),
    );
    // Free the context eagerly rather than waiting for GC; browsers cap how
    // many live WebGL contexts a page may hold.
    gl.getExtension('WEBGL_lose_context')?.loseContext();
    return SOFTWARE_RENDERER_RE.test(renderer);
  } catch {
    return false;
  }
}
