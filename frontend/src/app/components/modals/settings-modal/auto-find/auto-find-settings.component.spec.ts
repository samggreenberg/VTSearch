import { ComponentFixture, TestBed } from '@angular/core/testing';
import { Observable, of, throwError } from 'rxjs';

import {
  AutoFindExporterChange,
  AutoFindSettingsComponent,
} from './auto-find-settings.component';
import { ExportersApiService } from '../../../../services/exporters-api.service';
import type { ExporterEntry } from '../../../../generated/api-client/models/exporter-entry';
import { ImporterField } from '../../../../models/api.models';
import { provideZoneless } from '../../../../testing/zoneless-testbed';
import { settleZoneless } from '../../../../testing/settle-resource';

/**
 * Unit spec for the Auto-Find settings sub-panel. The panel loads the pickable
 * exporters (a tab strip whose active tab is the auto-export target), keeping
 * per-exporter field values that it emits to the parent. (Which detectors
 * auto-run is chosen on the Dashboard's Drafts/AutoRun tabs, not here.) The
 * exporters API service is stubbed so the init subscription resolves
 * synchronously.
 */
describe('AutoFindSettingsComponent', () => {
  let fixture: ComponentFixture<AutoFindSettingsComponent>;
  let component: AutoFindSettingsComponent;

  const exporters = [
    {
      name: 'server_json_file',
      display_name: 'JSON File',
      description: 'Write results to a JSON file',
      icon: 'download',
      hidden_from_picker: false,
      supported_payloads: ['find_results', 'labelset'],
      ui_mode: 'form',
      fields: [
        { key: 'filepath', field_type: 'text', label: 'File path', default: '/out.json' },
        { key: 'mode', field_type: 'select', label: 'Mode', options: ['a', 'b'] },
      ] as ImporterField[],
    },
    {
      name: 'email',
      display_name: 'Email',
      description: 'Email the results',
      icon: 'mail',
      hidden_from_picker: false,
      supported_payloads: ['find_results', 'labelset'],
      ui_mode: 'form',
      fields: [] as ImporterField[],
    },
    {
      name: 'secret_internal',
      display_name: 'Internal',
      description: 'hidden',
      icon: '',
      hidden_from_picker: true,
      supported_payloads: ['find_results'],
      ui_mode: 'form',
      fields: [] as ImporterField[],
    },
  ] as unknown as ExporterEntry[];

  /** An exporter whose destination list is only knowable at runtime: the
   *  shape issue #3360 reported as broken on this panel. */
  const dynamicExporter = {
    name: 'remote_queue',
    display_name: 'Remote Queue',
    description: 'Post results to a queue',
    icon: '',
    hidden_from_picker: false,
    supported_payloads: ['find_results'],
    ui_mode: 'form',
    fields: [
      { key: 'account', field_type: 'select', label: 'Account', options: ['personal', 'team'], default: 'personal' },
      {
        key: 'queue',
        field_type: 'select',
        label: 'Queue',
        required: true,
        dynamic_options: true,
        depends_on: ['account'],
      },
    ] as ImporterField[],
  } as unknown as ExporterEntry;

  let exportersResponse: () => Observable<unknown>;
  /** Every `(field, values)` pair the panel asked options for, in order. */
  let optionsCalls: { exporter: string; field: string; values: Record<string, string> }[];
  let optionsResponse: (
    exporter: string,
    field: string,
    values: Record<string, string>,
  ) => Observable<{ options: { value: string; label: string }[] }>;

  beforeEach(() => {
    exportersResponse = () => of(exporters);
    optionsCalls = [];
    optionsResponse = (_exporter, _field, values) =>
      of({
        options: [
          { value: `${values['account']}-a`, label: `${values['account']} A` },
          { value: `${values['account']}-b`, label: `${values['account']} B` },
        ],
      });

    const exportersStub: Partial<ExportersApiService> = {
      getExporters: () => exportersResponse() as Observable<never>,
      getFieldOptions: (exporter: string, field: string, values: Record<string, string>) => {
        optionsCalls.push({ exporter, field, values });
        return optionsResponse(exporter, field, values) as Observable<never>;
      },
    };

    TestBed.configureTestingModule({
      imports: [AutoFindSettingsComponent],
      providers: [
        ...provideZoneless(),
        { provide: ExportersApiService, useValue: exportersStub },
      ],
    });
  });

  async function create(inputs: {
    exporter?: string;
    fieldValues?: Record<string, Record<string, string>>;
  } = {}): Promise<void> {
    fixture = TestBed.createComponent(AutoFindSettingsComponent);
    component = fixture.componentInstance;
    fixture.componentRef.setInput('autofindExporter', inputs.exporter ?? '');
    fixture.componentRef.setInput('autofindExporterFieldValues', inputs.fieldValues ?? {});
    await settleZoneless(fixture);
  }

  function captureEmits(): AutoFindExporterChange[] {
    const out: AutoFindExporterChange[] = [];
    component.exporterChange.subscribe((v) => out.push(v));
    return out;
  }

  it('loads exporters, filtering out picker-hidden ones', async () => {
    await create();
    expect(component).toBeTruthy();
    expect(component.loadingExporters()).toBe(false);
    expect(component.exporters().map((e) => e.name)).toEqual(['server_json_file', 'email']);
  });

  it('points at the Dashboard AutoRun tab instead of hosting a detector checklist', async () => {
    await create();
    const text = (fixture.nativeElement as HTMLElement).textContent ?? '';
    expect(text).toContain('AutoRun');
    expect(text).not.toContain('Loading detectors…');
  });

  it('clears the "Loading…" placeholder in the DOM once the load settles (zoneless canary)', async () => {
    // Regression guard: the loading flag is written from an async subscribe
    // callback. As a plain property that write never schedules change
    // detection under zoneless, so the DOM stayed stuck on "Loading…" forever.
    // As a signal it repaints. settleZoneless() flushes only *scheduled* CD (it
    // does not force detectChanges), so a stale DOM here means a missing write.
    await create();
    const text = (fixture.nativeElement as HTMLElement).textContent ?? '';
    expect(text).not.toContain('Loading exporters…');
    // The real content rendered in place of the placeholder.
    expect(text).toContain('JSON File');
  });

  it('clears the exporter loading flag when the exporters call fails', async () => {
    exportersResponse = () => throwError(() => new Error('nope'));
    await create();
    expect(component.loadingExporters()).toBe(false);
    expect(component.exporters()).toEqual([]);
  });

  it('initialises the active exporter and cloned field values from inputs', async () => {
    const fv = { server_json_file: { filepath: '/saved.json' } };
    await create({ exporter: 'server_json_file', fieldValues: fv });
    expect(component.activeExporter).toBe('server_json_file');
    expect(component.fieldValue('filepath')).toBe('/saved.json');
    // Cloned, not aliased — editing the working copy must not touch the input.
    component.setFieldValue('filepath', '/edited.json');
    expect(fv.server_json_file.filepath).toBe('/saved.json');
  });

  it('reacts to input changes via the seeding effect', async () => {
    await create();
    expect(component.activeExporter).toBe('');

    fixture.componentRef.setInput('autofindExporter', 'email');
    fixture.componentRef.setInput('autofindExporterFieldValues', {
      email: { to: 'a@b.com' },
    });
    await settleZoneless(fixture);

    expect(component.activeExporter).toBe('email');
    expect(component.fieldValues).toEqual({ email: { to: 'a@b.com' } });
  });

  it('selectExporter seeds default field values on first pick and emits', async () => {
    await create();
    const emits = captureEmits();
    component.selectExporter('server_json_file');

    expect(component.activeExporter).toBe('server_json_file');
    // 'filepath' has a default; 'mode' does not, so only the default is seeded.
    expect(component.fieldValues['server_json_file']).toEqual({ filepath: '/out.json' });
    expect(emits).toEqual([
      { exporter: 'server_json_file', fieldValues: { server_json_file: { filepath: '/out.json' } } },
    ]);
  });

  it('selectExporter to the None tab emits an empty exporter', async () => {
    await create({ exporter: 'server_json_file' });
    const emits = captureEmits();
    component.selectExporter('');
    expect(component.activeExporter).toBe('');
    expect(emits.at(-1)?.exporter).toBe('');
  });

  it('does not overwrite existing field values when re-selecting an exporter', async () => {
    await create({ exporter: 'server_json_file', fieldValues: { server_json_file: { filepath: '/keep.json' } } });
    component.selectExporter('server_json_file');
    expect(component.fieldValues['server_json_file']).toEqual({ filepath: '/keep.json' });
  });

  it('activeFields returns the active exporter fields, or none for the None tab', async () => {
    await create({ exporter: 'server_json_file' });
    expect(component.activeFields.map((f) => f.key)).toEqual(['filepath', 'mode']);

    component.selectExporter('');
    expect(component.activeFields).toEqual([]);
  });

  it('setFieldValue writes the active exporter field and emits, ignoring writes with no active tab', async () => {
    await create({ exporter: 'server_json_file' });
    const emits = captureEmits();

    component.setFieldValue('filepath', '/new.json');
    expect(component.fieldValue('filepath')).toBe('/new.json');
    expect(emits.at(-1)).toEqual({
      exporter: 'server_json_file',
      fieldValues: { server_json_file: { filepath: '/new.json' } },
    });

    // On the None tab, writes are ignored (no emission).
    component.selectExporter('');
    const before = emits.length;
    component.setFieldValue('filepath', '/ignored.json');
    expect(emits.length).toBe(before);
  });

  it('inputType maps plugin field types to concrete input types', async () => {
    await create();
    expect(component.inputType({ field_type: 'password' } as ImporterField)).toBe('password');
    expect(component.inputType({ field_type: 'email' } as ImporterField)).toBe('email');
    expect(component.inputType({ field_type: 'url' } as ImporterField)).toBe('url');
    expect(component.inputType({ field_type: 'text' } as ImporterField)).toBe('text');
    // Double cast on purpose: `field_type` is a closed union in the generated
    // model, so this stands in for a *future* backend field type the frontend
    // doesn't map yet — which must still fall back to a plain text input.
    expect(component.inputType({ field_type: 'anything-else' } as unknown as ImporterField)).toBe(
      'text',
    );
  });

  describe('dynamic_options fields (issue #3360)', () => {
    beforeEach(() => {
      exportersResponse = () => of([...exporters, dynamicExporter]);
    });

    it('fetches the option list for the exporter restored from settings', async () => {
      await create({ exporter: 'remote_queue', fieldValues: { remote_queue: { account: 'team' } } });

      expect(optionsCalls).toEqual([
        { exporter: 'remote_queue', field: 'queue', values: { account: 'team' } },
      ]);
      expect(component.fieldOptions.optionsFor(component.activeFields[1])).toEqual([
        { value: 'team-a', label: 'team A' },
        { value: 'team-b', label: 'team B' },
      ]);
    });

    it('renders the fetched options rather than the (empty) declared ones', async () => {
      await create({ exporter: 'remote_queue', fieldValues: { remote_queue: { account: 'team' } } });

      const options = Array.from(
        (fixture.nativeElement as HTMLElement).querySelectorAll('select.form-select option'),
      ).map((o) => o.textContent?.trim());
      expect(options).toContain('team A');
      expect(options).toContain('team B');
    });

    it('persists the auto-selected option so the parent saves a real value', async () => {
      await create({ exporter: 'remote_queue', fieldValues: { remote_queue: { account: 'team' } } });
      // `queue` is required with no default: the helper picks the first option,
      // which is only useful if the panel emits it onward.
      expect(component.fieldValue('queue')).toBe('team-a');
      expect(component.fieldValues['remote_queue']).toEqual({ account: 'team', queue: 'team-a' });
    });

    it('fetches options when the exporter tab is selected', async () => {
      await create();
      component.selectExporter('remote_queue');
      expect(optionsCalls).toEqual([
        { exporter: 'remote_queue', field: 'queue', values: { account: 'personal' } },
      ]);
    });

    it('re-fetches a dependent field when the field it depends on changes', async () => {
      await create({ exporter: 'remote_queue', fieldValues: { remote_queue: { account: 'personal' } } });
      optionsCalls = [];

      component.setFieldValue('account', 'team');

      expect(optionsCalls).toEqual([
        { exporter: 'remote_queue', field: 'queue', values: { account: 'team', queue: '' } },
      ]);
      // The stale selection is replaced by one the new list actually offers.
      expect(component.fieldValue('queue')).toBe('team-a');
    });

    it('shows the fetch failure inline instead of an empty dropdown', async () => {
      // The 502 the options route returns when the plugin's remote service
      // fails, in the `flask_smorest.abort` envelope the app emits.
      optionsResponse = () =>
        throwError(() => ({ status: 502, error: { message: 'queue service down' } }));
      await create({ exporter: 'remote_queue' });

      expect(component.fieldOptions.error()['queue']).toContain('queue service down');
      const text = (fixture.nativeElement as HTMLElement).textContent ?? '';
      expect(text).toContain('queue service down');
    });

    it('drops the previous tab\'s options when switching exporters', async () => {
      await create({ exporter: 'remote_queue' });
      expect(component.fieldOptions.options()['queue']).toBeTruthy();

      component.selectExporter('server_json_file');
      expect(component.fieldOptions.options()['queue']).toBeUndefined();
    });
  });

  it('emits a cloned field-value map so the parent cannot mutate the working copy', async () => {
    await create({ exporter: 'server_json_file' });
    const emits = captureEmits();
    component.setFieldValue('filepath', '/x.json');
    const emitted = emits.at(-1)!.fieldValues;
    expect(emitted).not.toBe(component.fieldValues);
    expect(emitted['server_json_file']).not.toBe(component.fieldValues['server_json_file']);
  });
});
