import { ComponentFixture, TestBed } from '@angular/core/testing';
import { beforeEach, describe, expect, it } from 'vitest';

import { DropZoneComponent } from './drop-zone.component';
import { provideZoneless } from '../../testing/zoneless-testbed';

describe('DropZoneComponent', () => {
  let fixture: ComponentFixture<DropZoneComponent>;

  const zone = (): HTMLElement => fixture.nativeElement.querySelector('.drop-zone') as HTMLElement;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [DropZoneComponent],
      providers: [...provideZoneless()],
    }).compileComponents();
    fixture = TestBed.createComponent(DropZoneComponent);
    await fixture.whenStable();
  });

  it('shows the unanswered-required highlight only while the parent says so (#4311)', async () => {
    expect(zone().classList).not.toContain('drop-zone--missing');

    fixture.componentRef.setInput('missing', true);
    await fixture.whenStable();
    expect(zone().classList).toContain('drop-zone--missing');

    fixture.componentRef.setInput('missing', false);
    await fixture.whenStable();
    expect(zone().classList).not.toContain('drop-zone--missing');
  });
});
