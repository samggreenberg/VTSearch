import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ImageViewerComponent } from './image-viewer.component';
import { ToneMap } from '../face-even-out/even-out';
import { ActiveContextService } from '../../../services/active-context.service';
import { Media } from '../../../models/api.models';
import { configureZoneless } from '../../../testing/zoneless-testbed';
import { settleZoneless } from '../../../testing/settle-resource';

/**
 * The viewer's evening-out filter (#5674, #5675): one SVG component transfer
 * over the whole image, then greyscale, through the image's CSS `filter`. The
 * look itself is checked in a browser; jsdom draws nothing.
 */
describe('ImageViewerComponent evening out', () => {
  let fixture: ComponentFixture<ImageViewerComponent>;
  const face: Media = { id: 4, media_type: 'face', filename: 'f.jpg', md5: 'm', custom_metadata: {} };
  const MAP: ToneMap = {
    r: { amplitude: 0.9, exponent: 0.7, offset: 0.02 },
    g: { amplitude: 1, exponent: 0.7, offset: 0.02 },
    b: { amplitude: 1.1, exponent: 0.7, offset: 0.02 },
  };

  beforeEach(async () => {
    configureZoneless({ imports: [ImageViewerComponent], providers: [ActiveContextService] });
    fixture = TestBed.createComponent(ImageViewerComponent);
    fixture.componentRef.setInput('media', face);
    await settleZoneless(fixture);
  });

  afterEach(() => fixture.destroy());

  const img = (): HTMLImageElement => fixture.nativeElement.querySelector('img.image-element');
  const filter = (): SVGFilterElement | null => fixture.nativeElement.querySelector('svg.tone-filter filter');

  it('shows the image as it is with nothing to apply', () => {
    expect(filter()).toBeNull();
    expect(img().style.filter).toBe('');
  });

  it('draws the image through its own curve per channel, over the whole of it', async () => {
    fixture.componentRef.setInput('toneMap', MAP);
    await settleZoneless(fixture);
    const id = fixture.componentInstance.toneFilterId;
    expect(filter()!.id).toBe(id);
    // Fitted on the encoded values, so applied to them.
    expect(filter()!.getAttribute('color-interpolation-filters')).toBe('sRGB');
    const funcs = Array.from(filter()!.querySelectorAll('feComponentTransfer > *'));
    expect(funcs.map((f) => f.tagName)).toEqual(['feFuncR', 'feFuncG', 'feFuncB']);
    expect(funcs.map((f) => f.getAttribute('type'))).toEqual(['gamma', 'gamma', 'gamma']);
    expect(funcs[0].getAttribute('amplitude')).toBe('0.9');
    expect(funcs[2].getAttribute('amplitude')).toBe('1.1');
    expect(funcs[1].getAttribute('exponent')).toBe('0.7');
    expect(funcs[1].getAttribute('offset')).toBe('0.02');
    expect(img().style.filter).toBe(`url(#${id})`);

    fixture.componentRef.setInput('toneMap', null);
    await settleZoneless(fixture);
    expect(filter()).toBeNull();
    expect(img().style.filter).toBe('');
  });

  it('greys the image after its curve', async () => {
    fixture.componentRef.setInput('greyscale', true);
    await settleZoneless(fixture);
    expect(img().style.filter).toBe('grayscale(1)');

    fixture.componentRef.setInput('toneMap', MAP);
    await settleZoneless(fixture);
    expect(img().style.filter).toBe(`url(#${fixture.componentInstance.toneFilterId}) grayscale(1)`);
  });

  it('gives each viewer a filter id of its own', () => {
    const other = TestBed.createComponent(ImageViewerComponent);
    expect(other.componentInstance.toneFilterId).not.toBe(fixture.componentInstance.toneFilterId);
    other.destroy();
  });
});
