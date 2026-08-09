import { ComponentFixture, TestBed } from '@angular/core/testing';
import { Subject } from 'rxjs';

import { CombineDetectorsModalComponent } from './combine-detectors-modal.component';
import { DetectorsCrudApiService } from '../../../services/detectors-crud-api.service';
import type { DetectorCombineResponse } from '../../../generated/api-client/models/detector-combine-response';
import { DetectorRegistryEntry } from '../../../generated/api-client/models/detector-registry-entry';
import { configureZoneless } from '../../../testing/zoneless-testbed';
import { settleZoneless } from '../../../testing/settle-resource';

describe('CombineDetectorsModalComponent', () => {
  let component: CombineDetectorsModalComponent;
  let fixture: ComponentFixture<CombineDetectorsModalComponent>;
  /** Controllable stand-in for the combine request. */
  let combine$: Subject<DetectorCombineResponse>;

  const source = (name: string, numTraining: number): DetectorRegistryEntry =>
    ({ name, num_training: numTraining, media_type: 'audio' }) as DetectorRegistryEntry;

  beforeEach(async () => {
    combine$ = new Subject<DetectorCombineResponse>();

    await configureZoneless({
      imports: [CombineDetectorsModalComponent],
      providers: [
        {
          provide: DetectorsCrudApiService,
          useValue: { combine: () => combine$.asObservable() },
        },
      ],
    }).compileComponents();

    fixture = TestBed.createComponent(CombineDetectorsModalComponent);
    component = fixture.componentInstance;
    fixture.componentRef.setInput('sources', [source('Barks', 3), source('Yips', 5)]);
    fixture.componentRef.setInput('existingNames', ['Barks', 'Yips']);
  });

  const nameInput = () =>
    fixture.nativeElement.querySelector('#combine-new-name') as HTMLInputElement;
  const combineBtn = () =>
    fixture.nativeElement.querySelector('.btn--primary') as HTMLButtonElement;
  const cancelBtn = () =>
    fixture.nativeElement.querySelector('.btn--secondary') as HTMLButtonElement;
  const errorText = () => fixture.nativeElement.querySelector('.error-text') as HTMLElement | null;

  /** Type a name through ngModel and press Combine. */
  async function submitAs(name: string): Promise<void> {
    const el = nameInput();
    el.value = name;
    el.dispatchEvent(new Event('input'));
    await settleZoneless(fixture);
    combineBtn().click();
    await settleZoneless(fixture);
  }

  it('summarises the selected sources', async () => {
    await fixture.whenStable();
    const text = fixture.nativeElement.textContent as string;
    expect(text).toContain('Barks');
    expect(text).toContain('Yips');
    expect(text).toContain('2 detectors · 8 total labels');
  });

  it('blocks a name that collides with an existing detector', async () => {
    await fixture.whenStable();
    await submitAs('Barks');

    expect(errorText()?.textContent).toContain('already exists');
    expect(combineBtn().disabled).toBe(true);
  });

  it('disables both buttons while the combine is in flight', async () => {
    await fixture.whenStable();
    await submitAs('All Dog Sounds');

    expect(combineBtn().textContent).toContain('Combining…');
    expect(combineBtn().disabled).toBe(true);
    expect(cancelBtn().disabled).toBe(true);
  });

  // Zoneless staleness canary: the failure state is written from the combine
  // subscribe's error callback, and the success-only `created` output is the
  // component's sole other CD trigger. With plain fields the modal stayed on
  // "Combining…" with both buttons disabled and the server's message hidden
  // until an unrelated in-modal event happened to mark the view dirty.
  it('surfaces a rejected combine and re-enables the buttons (zoneless canary)', async () => {
    await fixture.whenStable();
    await submitAs('All Dog Sounds');

    combine$.error({ status: 409, error: { message: 'Name taken on disk.' } });
    await settleZoneless(fixture);

    expect(errorText()?.textContent).toContain('Name taken on disk.');
    expect(combineBtn().textContent).toContain('Combine');
    expect(combineBtn().disabled).toBe(false);
    expect(cancelBtn().disabled).toBe(false);
  });

  it('falls back to a status-specific message when the server sends no body', async () => {
    await fixture.whenStable();
    await submitAs('All Dog Sounds');

    combine$.error({ status: 422 });
    await settleZoneless(fixture);

    expect(errorText()?.textContent).toContain('Every label was a conflict');
  });

  it('emits the created name on success', async () => {
    await fixture.whenStable();
    let created = '';
    component.created.subscribe((n) => (created = n));
    await submitAs('All Dog Sounds');

    combine$.next({ name: 'All Dog Sounds' } as DetectorCombineResponse);
    await settleZoneless(fixture);

    expect(created).toBe('All Dog Sounds');
    expect(component.submitting()).toBe(false);
  });
});
