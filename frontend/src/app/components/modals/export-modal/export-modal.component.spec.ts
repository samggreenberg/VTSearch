import { ComponentFixture, TestBed } from '@angular/core/testing';

import { HttpTestingController } from '@angular/common/http/testing';
import { ExportModalComponent } from './export-modal.component';
import { ToastService } from '../../../services/toast.service';
import { provideZoneless } from '../../../testing/zoneless-testbed';
import { settleResource } from '../../../testing/settle-resource';
import { provideHttpTesting } from '../../../testing/test-providers';

describe('ExportModalComponent', () => {
  let component: ExportModalComponent;
  let fixture: ComponentFixture<ExportModalComponent>;
  let httpMock: HttpTestingController;

  const mockExporters = [
    { name: 'server_json_file', display_name: 'Server JSON', fields: [] },
    { name: 'hidden', display_name: 'Hidden', hidden_from_picker: true, fields: [] },
  ];
  const mockLabels = {
    labels: [
      { md5: 'a', label: 'good', filename: 'a.wav' },
      { md5: 'b', label: 'bad', filename: 'b.wav' },
      { md5: 'c', label: 'good', filename: 'c.wav', is_correction: true },
    ],
    available_columns: ['label', 'md5', 'filename', 'category', 'extra', 'name'],
  };

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [ExportModalComponent],
      providers: [...provideZoneless(), ...provideHttpTesting()],
    }).compileComponents();

    fixture = TestBed.createComponent(ExportModalComponent);
    component = fixture.componentInstance;
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => {
    httpMock.verify();
  });

  // The three init reads (dataset status, exporter list, labels) ride
  // `rxResource`, whose loaders run in a root effect rather than during
  // `detectChanges()`; tick to issue the GETs (the labels read also waits for
  // `ngOnInit` to set the input-derived filter), then settle before asserting.
  async function flushInit(): Promise<void> {
    // Zoneless + rxResource: TestBed.tick() runs ngOnInit and the resource
    // loader effects to issue the GETs. whenStable() can't be used — a loading
    // rxResource holds the app unstable — so the rxResource specs drive CD with
    // tick()/settleResource() and never call detectChanges().
    TestBed.tick();
    // The eager status/exporter GETs fire on the tick; the labels GET is
    // released by `ngOnInit`'s signal flip a microtask later, so settle first
    // to let all three become pending before flushing.
    await settleResource();
    httpMock.expectOne('/api/dataset/status').flush({ display_name: 'My Dataset' });
    httpMock.expectOne('/api/exporters').flush(mockExporters);
    httpMock.expectOne((r) => r.url === '/api/labels/export').flush(mockLabels);
    await settleResource();
  }

  it('should create', async () => {
    await flushInit();
    expect(component).toBeTruthy();
  });

  it('loads the exporter list, filtering hidden entries', async () => {
    await flushInit();
    expect(component.exporters().length).toBe(1);
    expect(component.exporters()[0].name).toBe('server_json_file');
  });

  it('builds columns from available_columns once labels resolve', async () => {
    await flushInit();
    expect(component.labelsLoaded()).toBe(true);
    const keys = component.columns.map((c) => c.key);
    expect(keys).toContain('extra'); // discovered metadata column
    expect(keys).not.toContain('origin'); // always-export keys stay out of the checkboxes
  });

  it('gives known metadata keys a curated label instead of the raw key', async () => {
    await flushInit();
    const nameCol = component.columns.find((c) => c.key === 'name');
    // The raw key stays for the export payload; only the checkbox label is curated.
    expect(nameCol?.label).toBe('Source');
    const extraCol = component.columns.find((c) => c.key === 'extra');
    expect(extraCol?.label).toBe('Extra'); // unknown keys fall back to title-casing
  });

  it('reports the tri-state column selection and toggles all on/off', async () => {
    await flushInit();
    expect(component.columnSelectionState).toBe('all'); // every column starts enabled

    component.columns[0].enabled = false;
    expect(component.columnSelectionState).toBe('some');

    component.toggleAllColumns(); // 'some' -> select all
    expect(component.columnSelectionState).toBe('all');
    expect(component.columns.every((c) => c.enabled)).toBe(true);

    component.toggleAllColumns(); // 'all' -> deselect all
    expect(component.columnSelectionState).toBe('none');
    expect(component.columns.every((c) => !c.enabled)).toBe(true);

    component.toggleAllColumns(); // 'none' -> select all
    expect(component.columnSelectionState).toBe('all');
  });

  it('copies exactly the checked columns, without the always-export keys', async () => {
    await flushInit();
    for (const col of component.columns) col.enabled = col.key === 'md5';

    // The clipboard flattens rows client-side, so `origin` could only ever
    // reach the paste as `[object Object]`; it stays out entirely (issue #2770).
    expect(component.clipboardColumns.map((c) => c.key)).toEqual(['md5']);
    expect(component.clipboardRows[0]).toEqual({ md5: 'a' });

    // Plugin exports still get them appended - there the exporter receives the
    // real dict and serializes it as JSON.
    component.startExporter(mockExporters[0] as never);
    const req = httpMock.expectOne('/api/exporters/export');
    expect(req.request.body.results.selected_columns).toEqual(['md5', 'origin', 'origin_name']);
    req.flush({ message: 'ok' });
  });

  it('slices the fetched labels by the active category', async () => {
    await flushInit();
    component.labelFilter = 'good';
    expect(component.filteredLabels.length).toBe(2);
    component.labelFilter = 'bad';
    expect(component.filteredLabels.length).toBe(1);
    component.labelFilter = 'corrections';
    expect(component.filteredLabels.length).toBe(1);
  });

  it('reports correction availability', async () => {
    await flushInit();
    expect(component.hasCorrections).toBe(true);
  });

  it('emits exported after a successful export run', async () => {
    await flushInit();
    vi.spyOn(component.exported, 'emit');
    // A fieldless exporter exports immediately.
    component.startExporter(mockExporters[0] as never);
    httpMock.expectOne('/api/exporters/export').flush({ success: true });
    expect(component.exported.emit).toHaveBeenCalled();
  });

  it('fires a success toast that outlives the closing modal', async () => {
    await flushInit();
    const toast = TestBed.inject(ToastService);
    const successSpy = vi.spyOn(toast, 'success');
    component.labelFilter = 'good'; // two matching labels in the fixture
    component.startExporter(mockExporters[0] as never);
    httpMock.expectOne('/api/exporters/export').flush({ success: true });
    expect(successSpy).toHaveBeenCalledTimes(1);
    expect(successSpy.mock.calls[0][0].message).toBe('Exported 2 labels to Server JSON');
  });

  it('seeds formValues from field defaults and carries field_values on the POST', async () => {
    await flushInit();
    // An exporter with a required field opens its form (rather than exporting
    // immediately) with each field's default seeded into `formValues`.
    const exporter = {
      name: 'server_json_file',
      display_name: 'Server JSON',
      fields: [
        {
          key: 'format',
          field_type: 'select',
          options: ['json', 'csv'],
          default: 'csv',
          required: true,
        },
      ],
    };
    component.startExporter(exporter as never);
    expect(component.selectedExporter).toBe(exporter);
    expect(component.formValues['format']).toBe('csv');

    // Submitting the form carries the seeded field values on the run-export POST.
    component.submitForm();
    const req = httpMock.expectOne('/api/exporters/export');
    expect(req.request.body.field_values).toEqual({ format: 'csv' });
    req.flush({ success: true });
  });

  it('does not auto-select the first option for a free-text combobox field', async () => {
    await flushInit();
    const exporter = {
      name: 'free_text_exporter',
      display_name: 'Free Text Exporter',
      fields: [
        {
          key: 'q',
          field_type: 'select',
          options: ['a', 'b'],
          allow_free_text: true,
        },
      ],
    };
    component.startExporter(exporter as never);
    expect(component.formValues['q']).toBe('');
  });

  it('emits closed on close', async () => {
    await flushInit();
    vi.spyOn(component.closed, 'emit');
    component.close();
    expect(component.closed.emit).toHaveBeenCalled();
  });

  describe('preview column resize', () => {
    /** Build a detached preview-style table and return its parts. */
    function makeTable(): { table: HTMLTableElement; th1: HTMLElement; handle: HTMLElement } {
      const table = document.createElement('table');
      const thead = document.createElement('thead');
      const tr = document.createElement('tr');
      const th1 = document.createElement('th');
      th1.setAttribute('data-col', 'label');
      const th2 = document.createElement('th');
      th2.setAttribute('data-col', 'md5');
      tr.append(th1, th2);
      thead.append(tr);
      table.append(thead);
      const handle = document.createElement('span');
      th1.append(handle);
      document.body.append(table);
      return { table, th1, handle };
    }

    function mousedownOn(handle: HTMLElement, clientX: number): MouseEvent {
      const ev = new MouseEvent('mousedown', { clientX });
      Object.defineProperty(ev, 'target', { value: handle });
      return ev;
    }

    it('freezes every column width and enters fixed layout on first grab', async () => {
      await flushInit();
      const { table, handle } = makeTable();
      component.startColResize(mousedownOn(handle, 100), 'label');
      expect(component.tableFixed).toBe(true);
      expect(component.colWidths).toHaveProperty('label');
      expect(component.colWidths).toHaveProperty('md5');
      expect(document.body.style.cursor).toBe('col-resize');
      component.onColResizeEnd();
      table.remove();
    });

    it('resizes the grabbed column by the drag delta, clamped to a minimum', async () => {
      await flushInit();
      const { table, handle } = makeTable();
      component.startColResize(mousedownOn(handle, 100), 'label');
      // offsetWidth is 0 under jsdom, so startWidth is 0; +80px drag → 80px.
      component.onColResizeMove(new MouseEvent('mousemove', { clientX: 180 }));
      expect(component.colWidths['label']).toBe(80);
      // A tiny drag can't shrink the column below the 40px floor.
      component.onColResizeMove(new MouseEvent('mousemove', { clientX: 110 }));
      expect(component.colWidths['label']).toBe(40);
      component.onColResizeEnd();
      table.remove();
    });

    it('ignores pointer motion once the drag ends and clears the cursor', async () => {
      await flushInit();
      const { table, handle } = makeTable();
      component.startColResize(mousedownOn(handle, 100), 'label');
      component.onColResizeMove(new MouseEvent('mousemove', { clientX: 180 }));
      component.onColResizeEnd();
      expect(document.body.style.cursor).toBe('');
      component.onColResizeMove(new MouseEvent('mousemove', { clientX: 400 }));
      expect(component.colWidths['label']).toBe(80); // unchanged after end
      table.remove();
    });
  });
});
