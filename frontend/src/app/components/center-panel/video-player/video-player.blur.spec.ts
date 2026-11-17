import { ComponentFixture, TestBed } from '@angular/core/testing';
import { Component, signal } from '@angular/core';
import { HttpTestingController, TestRequest } from '@angular/common/http/testing';
import { vi } from 'vitest';

import { VideoPlayerComponent } from './video-player.component';
import { BlurToggleComponent } from '../blur-toggle/blur-toggle.component';
import { BlurDirective } from '../../../directives/blur.directive';
import { ActiveContextService } from '../../../services/active-context.service';
import { SettingsStateService } from '../../../services/settings-state.service';
import { DATASET_OVERRIDE } from '../../../interceptors/active-context.interceptor';
import { Media } from '../../../models/api.models';
import type { AppSettings } from '../../../generated/api-client/models/app-settings';
import { BLUR_RADIUS_MAX_PX } from '../../../utils/blur-radius';
import { PEEK_DELAY_MS } from '../../../utils/focused-blur';
import { configureZoneless } from '../../../testing/zoneless-testbed';
import { provideHttpTesting } from '../../../testing/test-providers';
import { settleZoneless } from '../../../testing/settle-resource';
import { makeSettingsStateStub } from '../../../testing/mocks';

/**
 * The player, its Blur / Unblur button as the centre panel mounts it, and the
 * video's thumbnail in the list, which every list site draws through `vtBlur`.
 */
@Component({
  standalone: true,
  imports: [VideoPlayerComponent, BlurToggleComponent, BlurDirective],
  template: `
    <vt-video-player #v [media]="media()" [audioPlaying]="false" />
    @if (v.blurrable()) {
      <vt-blur-toggle [target]="v" noun="video" />
    }
    <img class="list-thumb" [vtBlur]="media().id" [src]="'/api/medias/' + media().id + '/thumbnail'" alt="" />
  `,
})
class HostComponent {
  readonly media = signal<Media>(clip(3));
}

function clip(id: number): Media {
  return { id, media_type: 'video', filename: `${id}.mp4`, md5: `m${id}`, custom_metadata: {} };
}

/**
 * Blur on the focused video (#5698): the video plays blurred when its state
 * says so, with the app's own control bar in place of the native controls a
 * blur would smear; the Blur / Unblur button flips that state for good, and
 * the thumbnail follows; and the peek, timed by the player as the image viewer
 * times its own. How the blur looks, and playback under it, are checked in a
 * browser; jsdom plays nothing.
 */
describe('VideoPlayerComponent blur', () => {
  let fixture: ComponentFixture<HostComponent>;
  let host: HostComponent;
  let httpMock: HttpTestingController;

  beforeEach(async () => {
    const settings = makeSettingsStateStub();
    // An instance with blur rules, so a set not read yet reads as blurred.
    settings.settings.set({ blur_rules: [{ field: 'nsfw', op: 'truthy' }] } as AppSettings);
    configureZoneless({
      imports: [HostComponent],
      providers: [...provideHttpTesting(), { provide: SettingsStateService, useValue: settings.stub }],
    });
    TestBed.inject(ActiveContextService).setActivePair('active-ds', 'det');
    httpMock = TestBed.inject(HttpTestingController);
    fixture = TestBed.createComponent(HostComponent);
    host = fixture.componentInstance;
    await settleZoneless(fixture);
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
    fixture.destroy();
    httpMock.verify();
  });

  const player = (): VideoPlayerComponent =>
    fixture.debugElement.query((d) => d.componentInstance instanceof VideoPlayerComponent).componentInstance;
  const video = (): HTMLVideoElement => fixture.nativeElement.querySelector('video');
  const frame = (): HTMLElement => fixture.nativeElement.querySelector('.video-frame');
  const bar = (): HTMLElement | null => fixture.nativeElement.querySelector('vt-video-controls');
  const thumb = (): HTMLImageElement => fixture.nativeElement.querySelector('img.list-thumb');
  const button = (): HTMLButtonElement | null => fixture.nativeElement.querySelector('[data-metric="videoBlur"]');

  /** The pending read of the active dataset's blurred set. */
  function blurredRead(): TestRequest {
    return httpMock.expectOne(
      (r) => r.url === '/api/medias/blurred' && r.context.get(DATASET_OVERRIDE) === 'active-ds',
    );
  }

  async function answerBlurred(ids: number[]): Promise<void> {
    blurredRead().flush({ ids });
    await settleZoneless(fixture);
  }

  function expectWrite(id: number, blurred: boolean): TestRequest {
    const req = httpMock.expectOne((r) => r.method === 'PUT' && r.url === `/api/medias/${id}/blur`);
    expect(req.request.context.get(DATASET_OVERRIDE)).toBe('active-ds');
    expect(req.request.body).toEqual({ blurred });
    return req;
  }

  /** Unblur the shown video (item 3), and let the write land. */
  async function unblurShown(): Promise<void> {
    player().toggleBlur();
    await settleZoneless(fixture);
    expectWrite(3, false).flush({ ok: true, id: 3, blurred: false });
    await settleZoneless(fixture);
  }

  /** The video drawn blurred, under the app's control bar rather than the native one. */
  function expectBlurred(): void {
    expect(video().classList).toContain('video-blurred');
    expect(video().hasAttribute('controls')).toBe(false);
    expect(video().hasAttribute('disablepictureinpicture')).toBe(true);
    expect(bar()).not.toBeNull();
    expect(thumb().classList).toContain('vt-blurred');
    expect(button()!.textContent!.trim()).toBe('Unblur');
    expect(button()!.getAttribute('aria-pressed')).toBe('true');
  }

  /** The video drawn sharp, under the native controls. */
  function expectSharp(): void {
    expect(video().classList).not.toContain('video-blurred');
    expect(video().hasAttribute('controls')).toBe(true);
    expect(video().hasAttribute('disablepictureinpicture')).toBe(false);
    expect(bar()).toBeNull();
    expect(thumb().classList).not.toContain('vt-blurred');
    expect(button()!.textContent!.trim()).toBe('Blur');
    expect(button()!.getAttribute('aria-pressed')).toBe('false');
  }

  it('plays the video blurred while its dataset\'s set is still out', async () => {
    expectBlurred();
    await answerBlurred([]);
    expectSharp();
  });

  it('plays a video the rules blurred blurred, and a sharp one sharp', async () => {
    await answerBlurred([3]);
    expectBlurred();

    host.media.set(clip(4));
    await settleZoneless(fixture);
    expectSharp();
  });

  it('blurs a video no rule matched, for good, at a click, and its thumbnail follows', async () => {
    await answerBlurred([]);
    expectSharp();

    button()!.click();
    await settleZoneless(fixture);
    const write = expectWrite(3, true);
    // Every view follows at once; the server's answer then settles it.
    expectBlurred();
    write.flush({ ok: true, id: 3, blurred: true });
    await settleZoneless(fixture);
    expectBlurred();
  });

  it('unblurs a video the rules blurred, and its thumbnail follows', async () => {
    await answerBlurred([3]);
    expectBlurred();

    await unblurShown();
    expectSharp();
  });

  it('says it is a video on the button', async () => {
    await answerBlurred([]);
    expect(button()!.title).toBe('Blur: hide this video behind a blur, everywhere it appears and for everyone (B)');
  });

  it('sizes the blur to the most any site gets until the element is measured', async () => {
    await answerBlurred([3]);
    expect(video().style.getPropertyValue('--video-blur-radius')).toBe(`${BLUR_RADIUS_MAX_PX}px`);
    await unblurShown();
    expect(video().style.getPropertyValue('--video-blur-radius')).toBe('');
  });

  it('plays or pauses at a click on the blurred picture, as the hidden native controls would', async () => {
    const play = vi.spyOn(HTMLMediaElement.prototype, 'play').mockResolvedValue(undefined);
    await answerBlurred([3]);
    video().click();
    expect(play).toHaveBeenCalledTimes(1);

    // With the native controls back, the element answers the click itself.
    await unblurShown();
    video().click();
    expect(play).toHaveBeenCalledTimes(1);
  });

  describe('the peek', () => {
    async function rested(ms: number): Promise<void> {
      vi.advanceTimersByTime(ms);
      TestBed.tick();
    }

    function enter(): void {
      frame().dispatchEvent(new MouseEvent('mouseenter'));
      TestBed.tick();
    }

    beforeEach(async () => {
      await answerBlurred([3, 4]);
      player().onLoadedData();
      vi.useFakeTimers();
      TestBed.tick();
    });

    it('lifts the blur after a second of rest, and puts it back at once on leave', async () => {
      enter();
      await rested(PEEK_DELAY_MS - 1);
      expect(video().classList).not.toContain('video-peek');
      await rested(1);
      expect(video().classList).toContain('video-peek');
      // Still blurred: the peek is a fade over the blur, and the controls stay the app's.
      expectBlurred();

      frame().dispatchEvent(new MouseEvent('mouseleave'));
      TestBed.tick();
      expect(video().classList).not.toContain('video-peek');
    });

    it('keeps peeking while the pointer is on the control bar', async () => {
      enter();
      await rested(PEEK_DELAY_MS);
      // The bar is inside the frame, so moving onto it does not leave the frame.
      bar()!.dispatchEvent(new MouseEvent('mouseenter'));
      TestBed.tick();
      expect(video().classList).toContain('video-peek');
    });

    it('waits again for the next video, under a pointer that stays on the frame', async () => {
      enter();
      await rested(PEEK_DELAY_MS);
      expect(video().classList).toContain('video-peek');

      // An arrow-key vote brings the next item with the pointer left where it was.
      host.media.set(clip(4));
      TestBed.tick();
      expect(video().classList).not.toContain('video-peek');
      // No wait runs until the new video's first frame is on screen.
      await rested(PEEK_DELAY_MS * 2);
      expect(video().classList).not.toContain('video-peek');
      player().onLoadedData();
      TestBed.tick();
      await rested(PEEK_DELAY_MS - 1);
      expect(video().classList).not.toContain('video-peek');
      await rested(1);
      expect(video().classList).toContain('video-peek');
    });
  });
});
