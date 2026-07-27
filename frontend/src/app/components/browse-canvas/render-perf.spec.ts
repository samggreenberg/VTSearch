import { afterEach, describe, expect, it, vi } from 'vitest';

import {
  detectSoftwareRenderer,
  FramePerfMonitor,
  SLOW_FRAME_ENTER_MS,
  SLOW_FRAME_MIN_SAMPLES,
  SLOW_FRAME_SAMPLE_CAP_MS,
} from './render-perf';

/**
 * Unit coverage for the software-rendering detection behind the browse
 * canvas's low-effects mode (issue #2695): the latching frame-cost monitor
 * and the WebGL renderer-string probe.
 */
describe('FramePerfMonitor', () => {
  it('starts fast by default', () => {
    expect(new FramePerfMonitor().slow).toBe(false);
  });

  it('honours the initiallySlow seed (a detected software renderer)', () => {
    expect(new FramePerfMonitor(true).slow).toBe(true);
  });

  it('never un-latches: fast frames after the seed leave it slow', () => {
    const m = new FramePerfMonitor(true);
    for (let i = 0; i < 50; i++) m.record(1);
    expect(m.slow).toBe(true);
  });

  it('does not judge before the minimum sample count, however slow', () => {
    const m = new FramePerfMonitor();
    for (let i = 0; i < SLOW_FRAME_MIN_SAMPLES - 1; i++) m.record(SLOW_FRAME_SAMPLE_CAP_MS);
    expect(m.slow).toBe(false);
  });

  it('latches on sustained slow frames', () => {
    const m = new FramePerfMonitor();
    for (let i = 0; i < SLOW_FRAME_MIN_SAMPLES; i++) m.record(SLOW_FRAME_ENTER_MS * 2);
    expect(m.slow).toBe(true);
  });

  it('stays latched once slow, even after the frames turn cheap (the degraded pipeline)', () => {
    const m = new FramePerfMonitor();
    for (let i = 0; i < 20; i++) m.record(60);
    expect(m.slow).toBe(true);
    for (let i = 0; i < 100; i++) m.record(2);
    expect(m.slow).toBe(true);
  });

  it('shrugs off a single huge hitch on an otherwise fast machine', () => {
    const m = new FramePerfMonitor();
    for (let i = 0; i < 30; i++) m.record(4);
    // One monster frame (tab backgrounded, GC pause) — capped and absorbed.
    m.record(5000);
    expect(m.slow).toBe(false);
    for (let i = 0; i < 10; i++) m.record(4);
    expect(m.slow).toBe(false);
  });

  it('keeps a healthy machine fast', () => {
    const m = new FramePerfMonitor();
    for (let i = 0; i < 200; i++) m.record(6);
    expect(m.slow).toBe(false);
  });
});

describe('detectSoftwareRenderer', () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  /** Stub `canvas.getContext` to return a minimal WebGL-shaped object whose
   * debug extension reports `rendererString`. */
  function stubWebGl(rendererString: string): void {
    const gl = {
      getExtension: (name: string) =>
        name === 'WEBGL_debug_renderer_info' ? { UNMASKED_RENDERER_WEBGL: 0x9246 } : null,
      getParameter: () => rendererString,
      RENDERER: 0x1f01,
    };
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(
      gl as unknown as RenderingContext,
    );
  }

  it('reports false when WebGL is unavailable (unknown, defer to frame timing)', () => {
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(null);
    expect(detectSoftwareRenderer()).toBe(false);
  });

  it('recognizes SwiftShader (Chrome with hardware acceleration off)', () => {
    stubWebGl('Google SwiftShader');
    expect(detectSoftwareRenderer()).toBe(true);
  });

  it('recognizes llvmpipe (Mesa software rasterizer)', () => {
    stubWebGl('Mesa/X.org, llvmpipe (LLVM 15.0.7, 256 bits)');
    expect(detectSoftwareRenderer()).toBe(true);
  });

  it('reports false for a hardware renderer string', () => {
    stubWebGl('ANGLE (NVIDIA, NVIDIA GeForce RTX 3080 Direct3D11 vs_5_0 ps_5_0, D3D11)');
    expect(detectSoftwareRenderer()).toBe(false);
  });

  it('reports false when the probe throws', () => {
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockImplementation(() => {
      throw new Error('blocked');
    });
    expect(detectSoftwareRenderer()).toBe(false);
  });
});
