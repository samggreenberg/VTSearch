import { ComponentFixture, TestBed } from '@angular/core/testing';

import { HttpTestingController } from '@angular/common/http/testing';
import { DetectorStatsModalComponent } from './detector-stats-modal.component';
import { configureZoneless } from '../../../testing/zoneless-testbed';
import { settleZoneless } from '../../../testing/settle-resource';
import { provideHttpTesting } from '../../../testing/test-providers';
import { wireVerdict } from '../../../testing/line-test';

describe('DetectorStatsModalComponent', () => {
  let component: DetectorStatsModalComponent;
  let fixture: ComponentFixture<DetectorStatsModalComponent>;
  let httpMock: HttpTestingController;

  const mockStats = {
    num_positive: 12,
    num_positive_resolved: 10,
    num_negative: 8,
    num_total: 20,
    active_dataset_name: 'My Dataset',
    media_type: 'audio',
    embedder: 'clap',
    embedder_display_name: 'CLAP (general, faster)',
    clipper: 'whole',
    text_query: '',
    media_example: '',
    created_at: 1700000000,
    last_trained_at: 1700000100,
    created_by: 'sam',
    autofind: false,
    readers: [],
    test_verdicts: [] as ReturnType<typeof wireVerdict>[],
    imported: null as { importer: string; params: Record<string, string>; imported_at: number } | null,
    importer_display_name: '',
    imported_rows: [] as { key: string; label: string; value: string; description: string }[],
  };

  beforeEach(async () => {
    await configureZoneless({
      imports: [DetectorStatsModalComponent],
      providers: [...provideHttpTesting()],
    }).compileComponents();

    fixture = TestBed.createComponent(DetectorStatsModalComponent);
    component = fixture.componentInstance;
    fixture.componentRef.setInput('detectorId', 'det1');
    fixture.componentRef.setInput('detectorName', 'My Detector');
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => {
    httpMock.verify();
  });

  it('should create', async () => {
    await fixture.whenStable();
    httpMock.expectOne('/api/detectors/registry/det1/stats').flush(mockStats);
    await settleZoneless(fixture);
    expect(component).toBeTruthy();
  });

  // Zoneless staleness canary: the stats land in an HTTP subscribe (an unpatched
  // callback) and repaint only because `loading`/`stats` are signals read in the
  // template. Flush the GET and assert the loaded table renders with no manual
  // `detectChanges`.
  it('repaints from loading to the loaded table (zoneless canary)', async () => {
    await fixture.whenStable();
    expect(fixture.nativeElement.querySelector('.loading-text')).toBeTruthy();

    httpMock.expectOne('/api/detectors/registry/det1/stats').flush(mockStats);
    await settleZoneless(fixture);

    expect(fixture.nativeElement.querySelector('.loading-text')).toBeFalsy();
    const text = fixture.nativeElement.textContent as string;
    expect(text).toContain('10 of 12 in "My Dataset"'); // resolvedSummary
  });

  it('says Untested when the detector keeps no test verdict', async () => {
    await fixture.whenStable();
    httpMock.expectOne('/api/detectors/registry/det1/stats').flush(mockStats);
    await settleZoneless(fixture);

    expect(fixture.nativeElement.querySelector('.verdict-list')).toBeFalsy();
    expect(fixture.nativeElement.querySelector('.verdict-empty').textContent).toContain('Untested');
  });

  it('lists the kept verdicts under Tested on, with a stale mark where it applies (#4526)', async () => {
    await fixture.whenStable();
    httpMock.expectOne('/api/detectors/registry/det1/stats').flush({
      ...mockStats,
      test_verdicts: [wireVerdict(), wireVerdict({ dataset_id: 'ds-old', dataset_name: 'drawings-old', stale: true, labelled: 1 })],
    });
    await settleZoneless(fixture);

    const rows = Array.from(fixture.nativeElement.querySelectorAll('.verdict')) as HTMLElement[];
    expect(rows.length).toBe(2);
    expect(rows[0].textContent).toContain('drawings-new: likely 78 ± 8% right, about half of them found (34 picks, 2026-10-05)');
    expect(rows[0].querySelector('.stale-mark')).toBeFalsy();
    expect(rows[1].classList).toContain('verdict--stale');
    expect(rows[1].textContent).toContain('(1 pick, 2026-10-05)');
    expect(rows[1].querySelector('.stale-mark')!.textContent).toContain('out of date');
    expect(rows[1].title).toContain('retrained since');
  });

  describe('the Import section (#5334)', () => {
    async function load(stats: typeof mockStats): Promise<HTMLElement> {
      await fixture.whenStable();
      httpMock.expectOne('/api/detectors/registry/det1/stats').flush(stats);
      await settleZoneless(fixture);
      return fixture.nativeElement as HTMLElement;
    }

    function sectionTitles(el: HTMLElement): (string | undefined)[] {
      return Array.from(el.querySelectorAll('.section-title')).map((h) => h.textContent?.trim());
    }

    const imported = { importer: 'server_json_file', params: { filepath: '/data/labels/cats.json' }, imported_at: 1700000000 };

    function rowTexts(el: HTMLElement): string[] {
      return Array.from(el.querySelectorAll('tr')).map((tr) =>
        Array.from(tr.querySelectorAll('td'))
          .map((td) => td.textContent?.trim())
          .join(' '),
      );
    }

    // Issue #5620: the rows read in the import form's words, as the server
    // words them, not as the record spells its keys.
    it('names the importer and its fields the way the import form does, and when', async () => {
      const el = await load({
        ...mockStats,
        imported,
        importer_display_name: 'Server JSON File',
        imported_rows: [
          { key: 'filepath', label: 'Server path', value: '/data/labels/cats.json', description: 'A labels file on the server.' },
        ],
      });

      expect(sectionTitles(el)).toContain('Import');
      const rows = rowTexts(el);
      expect(rows).toContain('Importer Server JSON File');
      expect(rows).toContain('Server path /data/labels/cats.json');
      expect(rows).not.toContain('Importer server_json_file');
      expect(rows.some((r) => r.startsWith('Imported ') && r !== 'Imported None')).toBe(true);
      const label = Array.from(el.querySelectorAll('.stat-label') as NodeListOf<HTMLElement>).find(
        (td) => td.textContent?.trim() === 'Server path',
      )!;
      expect(label.title).toBe('A labels file on the server.');
    });

    it('falls back to the recorded name and key for an importer that is no longer registered', async () => {
      const el = await load({
        ...mockStats,
        imported: { ...imported, importer: 'gone' },
        importer_display_name: '',
        imported_rows: [{ key: 'filepath', label: 'Filepath', value: '/data/labels/cats.json', description: '' }],
      });

      expect(rowTexts(el)).toContain('Importer gone');
      const label = Array.from(el.querySelectorAll('.stat-label') as NodeListOf<HTMLElement>).find(
        (td) => td.textContent?.trim() === 'Filepath',
      )!;
      expect(label.title).toBe('Importer field: filepath');
    });

    it('is absent for a detector no importer created', async () => {
      const el = await load(mockStats);
      expect(sectionTitles(el)).not.toContain('Import');
    });
  });

  // Issue #5617: a stat with no value says so in a word, not a dash.
  describe('empty values', () => {
    async function rowsOf(overrides: Partial<typeof mockStats>): Promise<string[]> {
      await fixture.whenStable();
      httpMock.expectOne('/api/detectors/registry/det1/stats').flush({ ...mockStats, ...overrides });
      await settleZoneless(fixture);
      return Array.from(fixture.nativeElement.querySelectorAll('tr') as NodeListOf<HTMLElement>).map((tr) =>
        Array.from(tr.querySelectorAll('td'))
          .map((td) => td.textContent?.trim())
          .join(' '),
      );
    }

    it('names the embedder by its display name', async () => {
      expect(await rowsOf({})).toContain('Embedder CLAP (general, faster)');
    });

    it('says None for an unset clipper, embedder or creator, and Never for an untrained detector', async () => {
      const rows = await rowsOf({
        clipper: '',
        embedder: '',
        embedder_display_name: '',
        created_by: '',
        last_trained_at: null as unknown as number,
      });
      expect(rows).toContain('Clipper None');
      expect(rows).toContain('Embedder None');
      expect(rows).toContain('Created by None');
      expect(rows).toContain('Last trained Never');
      expect(rows.some((r) => r.endsWith(' -'))).toBe(false);
    });
  });

  describe('the Media type row on a server locked to one media type (#5106)', () => {
    async function statLabelsWith(solo: string): Promise<(string | undefined)[]> {
      fixture.componentRef.setInput('soloMediaType', solo);
      await fixture.whenStable();
      httpMock.expectOne('/api/detectors/registry/det1/stats').flush(mockStats);
      await settleZoneless(fixture);
      return Array.from(fixture.nativeElement.querySelectorAll('.stat-label') as NodeListOf<HTMLElement>).map((el) =>
        el.textContent?.trim(),
      );
    }

    it('hides it for a detector of the locked type, where it could be nothing else', async () => {
      const labels = await statLabelsWith('audio');
      expect(labels).not.toContain('Media type');
      expect(labels).toContain('Embedder');
    });

    it('keeps it for a detector of a type the lock does not account for', async () => {
      expect(await statLabelsWith('image')).toContain('Media type');
    });
  });

  it('repaints the error text on a failed load (zoneless canary)', async () => {
    await fixture.whenStable();
    httpMock.expectOne('/api/detectors/registry/det1/stats').flush(
      { message: 'gone' },
      { status: 404, statusText: 'Not Found' },
    );
    await settleZoneless(fixture);

    const err = fixture.nativeElement.querySelector('.error-text') as HTMLElement;
    expect(err).toBeTruthy();
    expect(err.textContent).toContain('gone');
  });
});
