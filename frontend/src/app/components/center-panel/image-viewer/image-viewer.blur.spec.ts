import { ComponentFixture, TestBed } from '@angular/core/testing';
import { Component, signal } from '@angular/core';
import { HttpTestingController, TestRequest } from '@angular/common/http/testing';
import { vi } from 'vitest';

import { ImageViewerComponent, PEEK_DELAY_MS } from './image-viewer.component';
import { ImageViewControlsComponent } from '../image-view-controls/image-view-controls.component';
import { ActiveContextService } from '../../../services/active-context.service';
import { SettingsStateService } from '../../../services/settings-state.service';
import { DATASET_OVERRIDE } from '../../../interceptors/active-context.interceptor';
import { Media } from '../../../models/api.models';
import type { AppSettings } from '../../../generated/api-client/models/app-settings';
import { BLUR_RADIUS_MAX_PX, blurRadiusFor } from '../../../utils/blur-radius';
import { configureZoneless } from '../../../testing/zoneless-testbed';
import { provideHttpTesting } from '../../../testing/test-providers';
import { settleZoneless } from '../../../testing/settle-resource';
import { makeSettingsStateStub } from '../../../testing/mocks';

/** The viewer and its toolbar, as every host mounts them. */
@Component({
  standalone: true,
  imports: [ImageViewerComponent, ImageViewControlsComponent],
  template: `
    <vt-image-viewer #v [media]="media()" [datasetId]="datasetId()" [srcOverride]="srcOverride()" />
    <vt-image-view-controls [viewer]="v" />
  `,
})
class HostComponent {
  readonly media = signal<Media>(photo(4));
  readonly datasetId = signal('');
  readonly srcOverride = signal('');
}

function photo(id: number): Media {
  return { id, media_type: 'image', filename: `${id}.jpg`, md5: `m${id}`, custom_metadata: {} };
}

/**
 * Blur in the focused viewer (#5689): the image drawn blurred when its state
 * says so, the Blur / Unblur button that flips that state for good, and the
 * peek, which the viewer times itself so that a pointer left resting on the
 * pane never sees the next image sharp. How the blur looks, and the fade, are
 * checked in a browser; jsdom draws nothing.
 */
describe('ImageViewerComponent blur', () => {
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
    fixture.destroy();
    httpMock.verify();
  });

  const viewer = (): ImageViewerComponent =>
    fixture.debugElement.query((d) => d.componentInstance instanceof ImageViewerComponent).componentInstance;
  const img = (): HTMLImageElement => fixture.nativeElement.querySelector('img.image-element');
  const wrap = (): HTMLElement => fixture.nativeElement.querySelector('.image-wrap');
  const button = (): HTMLButtonElement | null => fixture.nativeElement.querySelector('[data-metric="imageBlur"]');

  /** The pending read of *datasetId*'s blurred set. */
  function blurredRead(datasetId: string): TestRequest {
    return httpMock.expectOne(
      (r) => r.url === '/api/medias/blurred' && r.context.get(DATASET_OVERRIDE) === datasetId,
    );
  }

  async function answerBlurred(datasetId: string, ids: number[]): Promise<void> {
    blurredRead(datasetId).flush({ ids });
    await settleZoneless(fixture);
  }

  function expectWrite(datasetId: string, id: number, blurred: boolean): TestRequest {
    const req = httpMock.expectOne((r) => r.method === 'PUT' && r.url === `/api/medias/${id}/blur`);
    expect(req.request.context.get(DATASET_OVERRIDE)).toBe(datasetId);
    expect(req.request.body).toEqual({ blurred });
    return req;
  }

  it('draws the image blurred while its dataset\'s set is still out', async () => {
    expect(img().classList).toContain('image-blurred');
    expect(button()!.textContent!.trim()).toBe('Unblur');

    await answerBlurred('active-ds', []);
    expect(img().classList).not.toContain('image-blurred');
    expect(button()!.textContent!.trim()).toBe('Blur');
  });

  it('blurs an image no rule matched, for good, at a click', async () => {
    await answerBlurred('active-ds', []);
    expect(button()!.getAttribute('aria-pressed')).toBe('false');

    button()!.click();
    await settleZoneless(fixture);
    const write = expectWrite('active-ds', 4, true);
    // Every view follows at once; the server's answer then settles it.
    expect(button()!.textContent!.trim()).toBe('Unblur');
    expect(button()!.getAttribute('aria-pressed')).toBe('true');
    expect(img().classList).toContain('image-blurred');
    write.flush({ ok: true, id: 4, blurred: true });
    await settleZoneless(fixture);
    expect(img().classList).toContain('image-blurred');
  });

  it('unblurs an image the rules blurred', async () => {
    await answerBlurred('active-ds', [4]);
    expect(img().classList).toContain('image-blurred');
    expect(button()!.textContent!.trim()).toBe('Unblur');

    button()!.click();
    await settleZoneless(fixture);
    expectWrite('active-ds', 4, false).flush({ ok: true, id: 4, blurred: false });
    await settleZoneless(fixture);
    expect(img().classList).not.toContain('image-blurred');
    expect(button()!.textContent!.trim()).toBe('Blur');
  });

  it('reads and writes the dataset its item is in, when that is not the active one', async () => {
    host.datasetId.set('photos');
    await settleZoneless(fixture);
    await answerBlurred('photos', []);
    // The active dataset's set was asked for first, and is not this item's.
    httpMock.match('/api/medias/blurred');

    viewer().toggleBlur();
    await settleZoneless(fixture);
    expectWrite('photos', 4, true).flush({ ok: true, id: 4, blurred: true });
    await settleZoneless(fixture);
    expect(img().classList).toContain('image-blurred');
  });

  it('leaves an uploaded example alone: no button, no blur', async () => {
    host.srcOverride.set('/api/server-media-files/example.jpg/image');
    await settleZoneless(fixture);
    httpMock.match('/api/medias/blurred');
    // Not an item of any dataset, so not blurred even before any set arrives.
    expect(img().classList).not.toContain('image-blurred');
    expect(button()).toBeNull();
    viewer().toggleBlur();
    httpMock.expectNone((r) => r.method === 'PUT');
  });

  it('sizes the blur to the picture as laid out, and to the most before that', async () => {
    await answerBlurred('active-ds', [4]);
    expect(img().style.getPropertyValue('--image-blur-radius')).toBe(`${BLUR_RADIUS_MAX_PX}px`);
    viewer().renderedW.set(500);
    viewer().renderedH.set(300);
    await settleZoneless(fixture);
    expect(img().style.getPropertyValue('--image-blur-radius')).toBe(`${blurRadiusFor(500, 300)}px`);
  });

  describe('the peek', () => {
    async function rested(ms: number): Promise<void> {
      vi.advanceTimersByTime(ms);
      TestBed.tick();
    }

    function enter(): void {
      wrap().dispatchEvent(new MouseEvent('mouseenter'));
      TestBed.tick();
    }

    beforeEach(async () => {
      await answerBlurred('active-ds', [4, 5]);
      viewer().onImageLoad();
      vi.useFakeTimers();
      TestBed.tick();
    });

    it('lifts the blur after a second of rest, and puts it back at once on leave', async () => {
      enter();
      await rested(PEEK_DELAY_MS - 1);
      expect(img().classList).not.toContain('image-peek');
      await rested(1);
      expect(img().classList).toContain('image-peek');
      // Still blurred: the peek is a fade over the blur, not an unblur.
      expect(img().classList).toContain('image-blurred');
      expect(button()!.textContent!.trim()).toBe('Unblur');

      wrap().dispatchEvent(new MouseEvent('mouseleave'));
      TestBed.tick();
      expect(img().classList).not.toContain('image-peek');
    });

    it('waits again for the next image, under a pointer that stays on the pane', async () => {
      enter();
      await rested(PEEK_DELAY_MS);
      expect(img().classList).toContain('image-peek');

      // An arrow-key vote brings the next item with the pointer left where it was.
      host.media.set(photo(5));
      TestBed.tick();
      expect(img().classList).not.toContain('image-peek');
      viewer().onImageLoad();
      TestBed.tick();
      await rested(PEEK_DELAY_MS - 1);
      expect(img().classList).not.toContain('image-peek');
      await rested(1);
      expect(img().classList).toContain('image-peek');
    });

    it('blurs at once when the image is blurred again under a resting pointer', async () => {
      viewer().toggleBlur();
      TestBed.tick();
      expectWrite('active-ds', 4, false).flush({ ok: true, id: 4, blurred: false });
      enter();
      await rested(PEEK_DELAY_MS * 3);
      expect(img().classList).not.toContain('image-blurred');

      // The B key, with the pointer still on the image.
      viewer().toggleBlur();
      TestBed.tick();
      expectWrite('active-ds', 4, true).flush({ ok: true, id: 4, blurred: true });
      TestBed.tick();
      expect(img().classList).toContain('image-blurred');
      expect(img().classList).not.toContain('image-peek');
      await rested(PEEK_DELAY_MS);
      expect(img().classList).toContain('image-peek');
    });
  });
});
