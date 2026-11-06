import { signal } from '@angular/core';
import { ComponentFixture, TestBed } from '@angular/core/testing';
import { HttpTestingController } from '@angular/common/http/testing';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { ReceivedDetectorsModalComponent } from './received-detectors-modal.component';
import { DetectorSharingFlowsService } from '../../../services/detector-sharing-flows.service';
import { SharingCapabilityService } from '../../../services/sharing-capability.service';
import { ToastService } from '../../../services/toast.service';
import type { DetectorRegistryEntry } from '../../../generated/api-client/models/detector-registry-entry';
import { configureZoneless } from '../../../testing/zoneless-testbed';
import { settleZoneless } from '../../../testing/settle-resource';
import { provideHttpTesting } from '../../../testing/test-providers';

const LIST_URL = '/api/detectors/sharing/received';
const RECEIVE_URL = '/api/detectors/sharing/receive';

/** Two containers shared with the user, one of them received already. */
const listing = [
  {
    remote_ref: 'c1',
    name: 'Dog barks',
    owner: 'alice',
    media_type: 'audio',
    record_count: 42,
    updated_at: 1793970000,
    shared_via: 'direct',
    detector_id: null,
  },
  {
    remote_ref: 'c2',
    name: 'Gulls',
    owner: 'bob',
    media_type: 'image',
    record_count: 7,
    updated_at: 1793960000,
    shared_via: 'group',
    detector_id: 'det-gulls',
  },
];

const receivedEntry = { id: 'det-barks', name: 'Dog barks', media_type: 'audio', sync_mode: 'linked' };

function applied(): object {
  return {
    ok: true,
    applied: true,
    message: "Received 'Dog barks' from alice.",
    integrity: { status: 'clean', count: 0 },
    remote_ref: 'c1',
    name: 'Dog barks',
    owner: 'alice',
    detector: receivedEntry,
  };
}

describe('ReceivedDetectorsModalComponent', () => {
  let fixture: ComponentFixture<ReceivedDetectorsModalComponent>;
  let httpMock: HttpTestingController;
  let received: DetectorRegistryEntry[];
  let shown: string[];

  const el = (): HTMLElement => fixture.nativeElement as HTMLElement;
  const rows = (): HTMLTableRowElement[] => Array.from(el().querySelectorAll('tbody tr'));
  const addButton = (): HTMLButtonElement => rows()[0].querySelector('.received-add') as HTMLButtonElement;

  async function open(items: object[] = listing): Promise<void> {
    fixture = TestBed.createComponent(ReceivedDetectorsModalComponent);
    received = [];
    shown = [];
    fixture.componentInstance.received.subscribe((e) => received.push(e));
    fixture.componentInstance.show.subscribe((id) => shown.push(id));
    await settleZoneless(fixture);
    httpMock.expectOne(LIST_URL).flush({ received: items });
    await settleZoneless(fixture);
  }

  beforeEach(async () => {
    await configureZoneless({
      imports: [ReceivedDetectorsModalComponent],
      providers: [
        ...provideHttpTesting(),
        { provide: SharingCapabilityService, useValue: { enabled: signal(true), displayName: signal('Shared folder') } },
      ],
    }).compileComponents();
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => httpMock.verify());

  it('lists each container: name, owner, type, labels, updated, and how it was shared', async () => {
    await open();
    const cells = rows().map((r) => Array.from(r.querySelectorAll('td')).map((td) => td.textContent!.trim()));
    expect(cells[0].slice(0, 4)).toEqual(['Dog barks', 'alice', 'audio', '42']);
    expect(cells[0][4]).toMatch(/^\d{4}-\d{2}-\d{2} \d{2}:\d{2}$/);
    expect(cells[0][5]).toBe('Directly');
    expect(cells[1][5]).toBe('By group');
    // The first can be added; the second is held already.
    expect(addButton().textContent!.trim()).toBe('Add');
    expect(rows()[1].querySelector('.received-add')).toBeNull();
    expect(rows()[1].querySelector('.received-added')!.textContent!.trim()).toBe('Added');
  });

  it('shows the copy already held', async () => {
    await open();
    (rows()[1].querySelector('.received-show') as HTMLButtonElement).click();
    expect(shown).toEqual(['det-gulls']);
  });

  it('receives a container, marks the row Added and hands the new detector on', async () => {
    const success = vi.spyOn(TestBed.inject(ToastService), 'success');
    await open();
    addButton().click();
    await settleZoneless(fixture);

    const req = httpMock.expectOne(RECEIVE_URL);
    expect(req.request.method).toBe('POST');
    expect(req.request.body).toEqual({ remote_ref: 'c1', foreign: 'abort' });
    req.flush(applied(), { status: 201, statusText: 'Created' });
    await settleZoneless(fixture);

    expect(received.map((e) => e.id)).toEqual(['det-barks']);
    expect(success).toHaveBeenCalledWith({ message: "Received 'Dog barks' from alice." });
    expect(rows()[0].querySelector('.received-added')).not.toBeNull();
    (rows()[0].querySelector('.received-show') as HTMLButtonElement).click();
    expect(shown).toEqual(['det-barks']);
  });

  it('asks about foreign records and re-sends the receive with the policy chosen', async () => {
    const ask = vi.spyOn(TestBed.inject(DetectorSharingFlowsService), 'askAboutForeignRecords').mockResolvedValue('skip');
    await open();
    addButton().click();
    await settleZoneless(fixture);

    httpMock.expectOne(RECEIVE_URL).flush({
      ok: true,
      applied: false,
      message: '3 votes were modified by someone other than alice.',
      integrity: { status: 'foreign_modifications', count: 3 },
      remote_ref: 'c1',
      name: 'Dog barks',
      owner: 'alice',
      detector: null,
    });
    await settleZoneless(fixture);
    expect(ask).toHaveBeenCalledWith({ id: 'c1', name: 'Dog barks' }, 3);

    const again = httpMock.expectOne(RECEIVE_URL);
    expect(again.request.body).toEqual({ remote_ref: 'c1', foreign: 'skip' });
    again.flush(applied(), { status: 201, statusText: 'Created' });
    await settleZoneless(fixture);
    expect(received.map((e) => e.id)).toEqual(['det-barks']);
  });

  it('receives nothing when the integrity dialog is cancelled', async () => {
    vi.spyOn(TestBed.inject(DetectorSharingFlowsService), 'askAboutForeignRecords').mockResolvedValue('cancel');
    await open();
    addButton().click();
    await settleZoneless(fixture);
    httpMock.expectOne(RECEIVE_URL).flush({
      ok: true,
      applied: false,
      message: '1 vote was modified by someone other than alice.',
      integrity: { status: 'foreign_modifications', count: 1 },
      remote_ref: 'c1',
      name: 'Dog barks',
      owner: 'alice',
      detector: null,
    });
    await settleZoneless(fixture);
    httpMock.expectNone(RECEIVE_URL);
    expect(received).toEqual([]);
    expect(addButton().disabled).toBe(false);
  });

  it('marks a container the user turns out to hold already', async () => {
    await open();
    addButton().click();
    await settleZoneless(fixture);
    httpMock
      .expectOne(RECEIVE_URL)
      .flush(
        { message: 'You already have this shared detector', error_code: 'already_received', detector_id: 'det-old' },
        { status: 409, statusText: 'Conflict' },
      );
    await settleZoneless(fixture);
    expect(received).toEqual([]);
    (rows()[0].querySelector('.received-show') as HTMLButtonElement).click();
    expect(shown).toEqual(['det-old']);
  });

  it('holds the other rows while one is being added', async () => {
    const two = [listing[0], { ...listing[1], remote_ref: 'c3', detector_id: null }];
    await open(two);
    addButton().click();
    await settleZoneless(fixture);
    expect(addButton().textContent!.trim()).toBe('Adding…');
    expect((rows()[1].querySelector('.received-add') as HTMLButtonElement).disabled).toBe(true);
    httpMock.expectOne(RECEIVE_URL).flush(applied(), { status: 201, statusText: 'Created' });
    await settleZoneless(fixture);
    expect((rows()[1].querySelector('.received-add') as HTMLButtonElement).disabled).toBe(false);
  });

  it('says when nothing is shared with the user, naming the source', async () => {
    await open([]);
    expect(rows()).toEqual([]);
    expect(el().querySelector('.received-empty')!.textContent).toContain('Shared folder');
  });

  it('shows a failed listing in place of the rows', async () => {
    fixture = TestBed.createComponent(ReceivedDetectorsModalComponent);
    await settleZoneless(fixture);
    httpMock
      .expectOne(LIST_URL)
      .flush({ message: 'The sharing service failed: timeout', error_code: 'sharing_failed' }, { status: 502, statusText: 'Bad Gateway' });
    await settleZoneless(fixture);
    expect(el().querySelector('.error-text')!.textContent).toContain('timeout');
  });
});
