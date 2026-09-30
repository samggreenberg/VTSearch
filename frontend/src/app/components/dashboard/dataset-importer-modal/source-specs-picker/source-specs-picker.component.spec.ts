import { ComponentFixture, TestBed } from '@angular/core/testing';

import { SourceSpecsPickerComponent } from './source-specs-picker.component';
import { provideZoneless } from '../../../../testing/zoneless-testbed';
import { settleZoneless } from '../../../../testing/settle-resource';
import { ClipperInfo, ConverterInfo } from '../../../../models/api.models';

describe('SourceSpecsPickerComponent', () => {
  let fixture: ComponentFixture<SourceSpecsPickerComponent>;

  const clippers: ClipperInfo[] = [
    { name: 'clip_default', display_name: 'Default clipper' } as ClipperInfo,
    { name: 'sliding', display_name: 'Sliding window' } as ClipperInfo,
  ];

  const converters: ConverterInfo[] = [
    {
      name: 'video2image',
      source_type: 'video',
      target_type: 'image',
      fields: [{ key: 'frames', label: 'Frames', field_type: 'number' }],
    } as ConverterInfo,
  ];

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [SourceSpecsPickerComponent],
      providers: [...provideZoneless()],
    }).compileComponents();

    fixture = TestBed.createComponent(SourceSpecsPickerComponent);
    fixture.componentRef.setInput('nativeType', 'image');
    fixture.componentRef.setInput('clippers', clippers);
    fixture.componentRef.setInput('availableConverters', converters);
    await settleZoneless(fixture);
  });

  // The picker's own stylesheet is encapsulated, so a class defined only in a
  // parent modal's SCSS never reaches these buttons: they rendered as bare
  // browser buttons (#4313). Their look has to come from the global `.link-btn`.
  it('styles both Details buttons with the global link button', () => {
    const buttons = Array.from(
      (fixture.nativeElement as HTMLElement).querySelectorAll<HTMLButtonElement>('.ssp-details-toggle'),
    );

    expect(buttons.map((b) => b.textContent?.trim())).toEqual(['Details ▸', 'Details ▸']);
    for (const b of buttons) {
      expect(b.classList.contains('link-btn')).toBe(true);
      expect(b.classList.contains('advanced-toggle')).toBe(false);
    }
  });
});
