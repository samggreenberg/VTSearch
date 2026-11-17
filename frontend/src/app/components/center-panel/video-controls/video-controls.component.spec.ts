import { ComponentFixture, TestBed } from '@angular/core/testing';
import { Component, signal } from '@angular/core';
import { vi } from 'vitest';

import { VideoControlsComponent, formatTime } from './video-controls.component';
import { ClipBounds } from '../../../utils/clip-window';
import { configureZoneless } from '../../../testing/zoneless-testbed';
import { settleZoneless } from '../../../testing/settle-resource';

@Component({
  standalone: true,
  imports: [VideoControlsComponent],
  template: `
    <video #v></video>
    <vt-video-controls [video]="v" [clip]="clip()" />
  `,
})
class HostComponent {
  readonly clip = signal<ClipBounds | null>(null);
}

/**
 * The control bar a blurred video plays under (#5698). jsdom has no media
 * pipeline, so the element's state is set by hand and its events dispatched,
 * as a browser would on playback.
 */
describe('VideoControlsComponent', () => {
  let fixture: ComponentFixture<HostComponent>;
  let video: HTMLVideoElement;
  // The element's state, behind writable stand-ins for jsdom's read-only fields.
  let state: { paused: boolean; currentTime: number; duration: number };

  beforeEach(async () => {
    configureZoneless({ imports: [HostComponent] });
    fixture = TestBed.createComponent(HostComponent);
    video = fixture.nativeElement.querySelector('video');
    state = { paused: true, currentTime: 0, duration: 75 };
    Object.defineProperty(video, 'paused', { configurable: true, get: () => state.paused });
    Object.defineProperty(video, 'duration', { configurable: true, get: () => state.duration });
    Object.defineProperty(video, 'currentTime', {
      configurable: true,
      get: () => state.currentTime,
      set: (v: number) => (state.currentTime = v),
    });
    await settleZoneless(fixture);
  });

  afterEach(() => {
    vi.restoreAllMocks();
    fixture.destroy();
  });

  const q = <T extends Element>(selector: string): T => fixture.nativeElement.querySelector(selector);
  const playButton = (): HTMLButtonElement => q('.vc-play');
  const muteButton = (): HTMLButtonElement => q('.vc-mute');
  const seek = (): HTMLInputElement => q('.vc-seek');
  const volume = (): HTMLInputElement => q('.vc-volume');
  const time = (): string => q('.vc-time').textContent!.trim();

  async function fire(type: string): Promise<void> {
    video.dispatchEvent(new Event(type));
    await settleZoneless(fixture);
  }

  function slide(input: HTMLInputElement, value: number): void {
    input.value = String(value);
    input.dispatchEvent(new Event('input'));
  }

  it('plays a paused video and pauses a playing one', async () => {
    const play = vi.spyOn(video, 'play').mockResolvedValue(undefined);
    const pause = vi.spyOn(video, 'pause');
    expect(playButton().getAttribute('aria-label')).toBe('Play');
    playButton().click();
    expect(play).toHaveBeenCalledTimes(1);

    state.paused = false;
    await fire('play');
    expect(playButton().getAttribute('aria-label')).toBe('Pause');
    playButton().click();
    expect(pause).toHaveBeenCalledTimes(1);
  });

  it('follows playback in the time and the seek bar', async () => {
    state.currentTime = 12;
    await fire('timeupdate');
    expect(time()).toBe('0:12 / 1:15');
    expect(seek().max).toBe('75');
    expect(Number(seek().value)).toBe(12);
  });

  it('seeks where the bar is moved to', async () => {
    slide(seek(), 40);
    expect(video.currentTime).toBe(40);
    await settleZoneless(fixture);
    expect(time()).toBe('0:40 / 1:15');
  });

  it('spans the clip window when the item has one', async () => {
    fixture.componentInstance.clip.set({ start: 10, end: 20 });
    state.currentTime = 12;
    await fire('timeupdate');
    expect(seek().min).toBe('10');
    expect(seek().max).toBe('20');
    expect(time()).toBe('0:02 / 0:10');
  });

  it('cannot seek before the length is known', async () => {
    state.duration = Number.NaN;
    await fire('emptied');
    expect(seek().disabled).toBe(true);
    expect(time()).toBe('0:00 / 0:00');
  });

  it('mutes and unmutes, and shows a muted video at no volume', async () => {
    muteButton().click();
    expect(video.muted).toBe(true);
    await fire('volumechange');
    expect(muteButton().getAttribute('aria-pressed')).toBe('true');
    expect(Number(volume().value)).toBe(0);

    muteButton().click();
    expect(video.muted).toBe(false);
    await fire('volumechange');
    expect(Number(volume().value)).toBe(1);
  });

  it('sets the volume from its slider, and raising it unmutes', async () => {
    video.muted = true;
    slide(volume(), 0.4);
    expect(video.volume).toBeCloseTo(0.4, 5);
    expect(video.muted).toBe(false);
  });

  it('follows a volume set elsewhere (Shift+arrow)', async () => {
    video.volume = 0.25;
    await fire('volumechange');
    expect(Number(volume().value)).toBeCloseTo(0.25, 5);
  });
});

describe('formatTime', () => {
  it('writes minutes and seconds, and hours from an hour', () => {
    expect(formatTime(0)).toBe('0:00');
    expect(formatTime(9.7)).toBe('0:09');
    expect(formatTime(75)).toBe('1:15');
    expect(formatTime(3725)).toBe('1:02:05');
  });

  it('writes nothing it does not know as zero', () => {
    expect(formatTime(Number.NaN)).toBe('0:00');
    expect(formatTime(-3)).toBe('0:00');
  });
});
