import { ComponentFixture, TestBed } from '@angular/core/testing';

import { HttpTestingController } from '@angular/common/http/testing';
import { DuplicatesModalComponent } from './duplicates-modal.component';
import { configureZoneless } from '../../../testing/zoneless-testbed';
import { settleZoneless } from '../../../testing/settle-resource';
import { provideHttpTesting } from '../../../testing/test-providers';

describe('DuplicatesModalComponent', () => {
  let component: DuplicatesModalComponent;
  let fixture: ComponentFixture<DuplicatesModalComponent>;
  let httpMock: HttpTestingController;

  const mockDupes = {
    duplicate_sets: [
      {
        name: 'a.wav',
        members: [
          { md5: 'm1', filename: 'a.wav', category: 'c1', origin_name: 'a.wav', importer: 'server_folder' },
          { md5: 'm1', filename: 'b.wav', category: 'c2', origin_name: 'b.wav', importer: 'http_archive' },
        ],
      },
      {
        name: 'x.txt',
        members: [
          { md5: 'm2', filename: 'x.txt', category: 'c1', origin_name: 'x.txt', importer: 'demo' },
          { md5: 'm3', filename: 'y.txt', category: 'c1', origin_name: 'y.txt', importer: 'demo' },
        ],
      },
    ],
  };

  beforeEach(async () => {
    await configureZoneless({
      imports: [DuplicatesModalComponent],
      providers: [...provideHttpTesting()],
    }).compileComponents();

    fixture = TestBed.createComponent(DuplicatesModalComponent);
    component = fixture.componentInstance;
    fixture.componentRef.setInput('datasetId', 'ds1');
    fixture.componentRef.setInput('datasetName', 'My Dataset');
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => {
    httpMock.verify();
  });

  async function flushInit(payload: object = mockDupes): Promise<void> {
    await fixture.whenStable();
    httpMock.expectOne('/api/datasets/registry/ds1/duplicates').flush(payload);
    await settleZoneless(fixture);
  }

  it('should create', async () => {
    await flushInit();
    expect(component).toBeTruthy();
  });

  it('flattens sets to one row per member, numbering sets 1-based', async () => {
    await flushInit();
    const rows = component.rows();
    expect(rows.length).toBe(4);
    expect(rows.map((r) => r.dupe_set)).toEqual(['1', '1', '2', '2']);
    expect(rows[1].filename).toBe('b.wav');
    expect(rows[1].importer).toBe('http_archive');
  });

  it('renders the table with a Dupe Set column and a separator between sets', async () => {
    await flushInit();
    const headers = Array.from(
      fixture.nativeElement.querySelectorAll('thead th') as NodeListOf<HTMLElement>,
    ).map((th) => th.textContent?.trim());
    expect(headers).toContain('Dupe Set');
    expect(headers).not.toContain('Label');
    // Separator sits on the first row of the second set (index 2), not within a set.
    expect(component.isSetStart(0)).toBe(false);
    expect(component.isSetStart(1)).toBe(false);
    expect(component.isSetStart(2)).toBe(true);
    expect(fixture.nativeElement.querySelectorAll('tbody tr.set-start').length).toBe(1);
  });

  it('drops a toggled-off column from the table and the clipboard payload', async () => {
    await flushInit();
    const md5Col = component.columns.find((c) => c.key === 'md5')!;
    md5Col.enabled = false;
    expect(component.enabledColumns.map((c) => c.key)).not.toContain('md5');
    expect(component.clipboardColumns.map((c) => c.key)).not.toContain('md5');
  });

  it('shows the empty state when there are no duplicate sets', async () => {
    await flushInit({ duplicate_sets: [] });
    expect(fixture.nativeElement.querySelector('.empty-state')).toBeTruthy();
    expect(fixture.nativeElement.querySelector('.dupes-table')).toBeFalsy();
  });

  it('surfaces the not-loaded 400 message from the backend', async () => {
    await fixture.whenStable();
    httpMock.expectOne('/api/datasets/registry/ds1/duplicates').flush(
      { message: 'Load the dataset to browse its duplicates' },
      { status: 400, statusText: 'Bad Request' },
    );
    await settleZoneless(fixture);
    const err = fixture.nativeElement.querySelector('.error-text') as HTMLElement;
    expect(err).toBeTruthy();
    expect(err.textContent).toContain('Load the dataset');
  });

  it('emits closed from the Back button', async () => {
    await flushInit();
    vi.spyOn(component.closed, 'emit');
    (fixture.nativeElement.querySelector('.back-btn') as HTMLButtonElement).click();
    expect(component.closed.emit).toHaveBeenCalled();
  });
});
