import { ElementRef } from '@angular/core';
import { ComponentFixture, TestBed } from '@angular/core/testing';

import { AudioCropOverlayComponent, AudioCropResult } from './audio-crop-overlay.component';
import { provideZoneless } from '../../../testing/zoneless-testbed';

/**
 * These specs exercise the crop geometry and emitted crop-region state without a
 * real browser. The waveform-decode path (`ngAfterViewInit` → `decodeAudioBuffer`)
 * is never triggered — we skip `detectChanges()` and instead hand the component a
 * fake `<canvas>` ref plus a known `duration`, so every pointer handler runs
 * against a deterministic 600px-wide, 10-second timeline. `draw()` is a no-op in
 * tests because the fake canvas's `getContext` returns null.
 */
describe('AudioCropOverlayComponent', () => {
  let component: AudioCropOverlayComponent;
  let fixture: ComponentFixture<AudioCropOverlayComponent>;
  let capture: ReturnType<typeof vi.fn>;
  let release: ReturnType<typeof vi.fn>;

  // 600px canvas mapped onto a 10s clip → 1px == 1/60s (x px == x/60 sec).
  const WIDTH = 600;
  const DURATION = 10;

  function ptr(clientX: number): PointerEvent {
    return {
      clientX,
      clientY: 0,
      pointerId: 1,
      preventDefault: () => {},
    } as unknown as PointerEvent;
  }

  beforeEach(() => {
    TestBed.configureTestingModule({
      imports: [AudioCropOverlayComponent],
      providers: [...provideZoneless()],
    });
    fixture = TestBed.createComponent(AudioCropOverlayComponent);
    component = fixture.componentInstance;

    capture = vi.fn();
    release = vi.fn();
    const canvas = {
      width: WIDTH,
      height: 120,
      getContext: () => null, // draw() short-circuits; no canvas rendering in jsdom.
      getBoundingClientRect: () =>
        ({ left: 0, top: 0, width: WIDTH, height: 120 }) as DOMRect,
      setPointerCapture: capture,
      releasePointerCapture: release,
    };
    // `canvasRef` is a signal query; stub it with a zero-arg function returning
    // the fake canvas so `this.canvasRef()` works without a real view/query
    // resolution (and `draw()` short-circuits on the null 2D context).
    (component as unknown as { canvasRef: () => ElementRef<HTMLCanvasElement> }).canvasRef =
      () => ({ nativeElement: canvas }) as unknown as ElementRef<HTMLCanvasElement>;

    // Pretend the waveform decoded to a 10s clip with the full range selected.
    component.duration = DURATION;
    component.start = 0;
    component.end = DURATION;
  });

  it('creates', () => {
    expect(component).toBeTruthy();
  });

  describe('formatTime', () => {
    it('renders minutes:seconds with two fractional digits, zero-padded', () => {
      expect(component.formatTime(0)).toBe('0:00.00');
      expect(component.formatTime(9.5)).toBe('0:09.50');
      expect(component.formatTime(65)).toBe('1:05.00');
      expect(component.formatTime(125.25)).toBe('2:05.25');
    });

    it('guards against non-finite input', () => {
      expect(component.formatTime(NaN)).toBe('0:00');
      expect(component.formatTime(Infinity)).toBe('0:00');
    });
  });

  describe('onAudioLoadedMetadata', () => {
    function metaEvent(duration: number): Event {
      return { target: { duration } } as unknown as Event;
    }

    it('seeds duration/end from the <audio> element when duration is still unknown', () => {
      component.duration = 0;
      component.end = 0;
      component.onAudioLoadedMetadata(metaEvent(7.5));
      expect(component.duration).toBe(7.5);
      expect(component.end).toBe(7.5);
    });

    it('does not override a duration already known from the decoded waveform', () => {
      component.onAudioLoadedMetadata(metaEvent(999));
      expect(component.duration).toBe(DURATION);
      expect(component.end).toBe(DURATION);
    });

    it('ignores a non-finite element duration (streaming/unknown length)', () => {
      component.duration = 0;
      component.onAudioLoadedMetadata(metaEvent(Infinity));
      expect(component.duration).toBe(0);
    });
  });

  describe('pointer dragging', () => {
    beforeEach(() => {
      // Start with a mid-clip selection: start=2s (x=120), end=8s (x=480).
      component.start = 2;
      component.end = 8;
    });

    it('drags the start handle, clamped to [0, end - 0.05]', () => {
      component.onPointerDown(ptr(120)); // grab start handle at x=120
      expect(capture).toHaveBeenCalledWith(1);
      component.onPointerMove(ptr(60)); // 1s
      expect(component.start).toBeCloseTo(1, 5);
      // Drag past the end handle: clamped to end - 0.05.
      component.onPointerMove(ptr(600));
      expect(component.start).toBeCloseTo(8 - 0.05, 5);
      component.onPointerUp(ptr(600));
      expect(release).toHaveBeenCalledWith(1);
    });

    it('drags the end handle, clamped to [start + 0.05, duration]', () => {
      component.onPointerDown(ptr(480)); // grab end handle at x=480
      component.onPointerMove(ptr(360)); // 6s
      expect(component.end).toBeCloseTo(6, 5);
      // Drag past the right edge: clamped to duration.
      component.onPointerMove(ptr(900));
      expect(component.end).toBeCloseTo(DURATION, 5);
    });

    it('moves the whole selection, preserving its width', () => {
      component.onPointerDown(ptr(300)); // inside selection → move mode
      component.onPointerMove(ptr(360)); // +1s shift
      expect(component.start).toBeCloseTo(3, 5);
      expect(component.end).toBeCloseTo(9, 5);
      // Width (6s) is preserved throughout.
      expect(component.end - component.start).toBeCloseTo(6, 5);
    });

    it('clamps a move so the selection cannot slide past the clip end', () => {
      component.onPointerDown(ptr(300)); // move mode, offset = 3s
      component.onPointerMove(ptr(600)); // would push start to 7s → end 13s
      expect(component.end).toBeCloseTo(DURATION, 5);
      expect(component.start).toBeCloseTo(DURATION - 6, 5); // 4s
    });

    it('clicking outside the selection repositions the nearer handle', () => {
      component.onPointerDown(ptr(540)); // right of end handle (x2=480)
      expect(component.end).toBeCloseTo(9, 5); // 540px → 9s
      expect(component.start).toBeCloseTo(2, 5); // start untouched
    });

    it('ignores pointer motion when no drag is in progress', () => {
      component.onPointerMove(ptr(300));
      expect(component.start).toBe(2);
      expect(component.end).toBe(8);
    });

    it('does not start a drag before the clip duration is known', () => {
      component.duration = 0;
      component.onPointerDown(ptr(300));
      expect(capture).not.toHaveBeenCalled();
    });
  });

  describe('apply / cancel', () => {
    it('emits the selected region on apply', () => {
      component.start = 1.5;
      component.end = 6.25;
      let result: AudioCropResult | undefined;
      component.applied.subscribe((r) => (result = r));
      component.apply();
      expect(result).toEqual({ start: 1.5, end: 6.25 });
    });

    it('does not emit when the region is empty or the clip has no duration', () => {
      const spy = vi.fn();
      component.applied.subscribe(spy);
      component.end = component.start; // empty selection
      component.apply();
      component.duration = 0;
      component.apply();
      expect(spy).not.toHaveBeenCalled();
    });

    it('emits cancelled on cancel', () => {
      const spy = vi.fn();
      component.cancelled.subscribe(spy);
      component.cancel();
      expect(spy).toHaveBeenCalledTimes(1);
    });
  });

  describe('computePeaks', () => {
    it('reduces channel samples to n bins holding each bin peak amplitude', () => {
      const samples = new Float32Array([0, 0.5, 0, 0, 0.2, -0.9, 0.1, 0, 0.3, 0, 0, 0]);
      const buffer = { getChannelData: () => samples } as unknown as AudioBuffer;
      // 12 samples / 3 bins → 4 samples per bin.
      const peaks = (
        component as unknown as { computePeaks(b: AudioBuffer, n: number): number[] }
      ).computePeaks(buffer, 3);
      expect(peaks).toHaveLength(3);
      expect(peaks[0]).toBeCloseTo(0.5, 5);
      expect(peaks[1]).toBeCloseTo(0.9, 5); // abs value → 0.9, not -0.9
      expect(peaks[2]).toBeCloseTo(0.3, 5);
    });
  });
});
