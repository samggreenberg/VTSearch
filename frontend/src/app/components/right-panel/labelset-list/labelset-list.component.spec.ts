import { ComponentFixture, TestBed } from '@angular/core/testing';

import { LabelsetListComponent } from './labelset-list.component';
import type { DetectorLabelView } from '../../../generated/api-client/models/detector-label-view';
import { ActiveContextService } from '../../../services/active-context.service';
import { MediaTypeCapabilityService } from '../../../services/media-type-capability.service';
import { provideZoneless } from '../../../testing/zoneless-testbed';
import { settleZoneless } from '../../../testing/settle-resource';
import { blurKey, provideBlurStateStub } from '../../../testing/blur-state-stub';

function label(id: string, over: Partial<DetectorLabelView> = {}): DetectorLabelView {
  return {
    id,
    label: 'good',
    media_type: 'image',
    name: `${id}.png`,
    filename: `${id}.png`,
    origin_name: `${id}.png`,
    md5: id,
    cid: null,
    time: -1,
    score: -1,
    region_box: null,
    blurred: null,
    ...over,
  };
}

describe('LabelsetListComponent', () => {
  let fixture: ComponentFixture<LabelsetListComponent>;
  let blur: ReturnType<typeof provideBlurStateStub>;

  beforeEach(async () => {
    blur = provideBlurStateStub();
    await TestBed.configureTestingModule({
      imports: [LabelsetListComponent],
      providers: [
        ...provideZoneless(),
        blur.provider,
        { provide: MediaTypeCapabilityService, useValue: { usesThumbnails: () => true } },
      ],
    }).compileComponents();
    TestBed.inject(ActiveContextService).setActivePair('ds-1', 'det-1');
    fixture = TestBed.createComponent(LabelsetListComponent);
    fixture.componentRef.setInput('modelName', 'my-detector');
  });

  /** Whether each stored label's tile is blurred, by label name. */
  function blurredByName(): Record<string, boolean> {
    const imgs = Array.from((fixture.nativeElement as HTMLElement).querySelectorAll('img.vote-thumbnail'));
    return Object.fromEntries(imgs.map((img) => [img.getAttribute('alt'), img.classList.contains('vt-blurred')]));
  }

  it('blurs a stored label in the loaded dataset by that dataset’s state for its cid (#5696)', async () => {
    blur.blurred.set(new Set([blurKey('ds-1', 11)]));
    fixture.componentRef.setInput('elements', [label('hot', { cid: 11 }), label('calm', { cid: 12 })]);
    await settleZoneless(fixture);
    expect(blurredByName()).toEqual({ 'hot.png': true, 'calm.png': false });
  });

  it('blurs a stored label outside the loaded dataset by the server’s answer', async () => {
    // The dataset state is never consulted for it, even for a colliding id.
    blur.blurred.set(new Set([blurKey('ds-1', 0)]));
    fixture.componentRef.setInput('elements', [
      label('hot', { blurred: true }),
      label('calm', { blurred: false }),
      label('unsaid'),
    ]);
    await settleZoneless(fixture);
    expect(blurredByName()).toEqual({ 'hot.png': true, 'calm.png': false, 'unsaid.png': false });
  });
});
