import { ComponentFixture, TestBed } from '@angular/core/testing';
import { Observable, of, throwError } from 'rxjs';

import {
  AutoFindExporterChange,
  AutoFindSettingsComponent,
} from './auto-find-settings.component';
import { DetectorsRegistryApiService } from '../../../../services/detectors-registry-api.service';
import { ExportersApiService } from '../../../../services/exporters-api.service';
import type { ExporterEntry } from '../../../../generated/api-client/models/exporter-entry';
import { ImporterField } from '../../../../models/api.models';
import { provideZoneless } from '../../../../testing/zoneless-testbed';
import { settleZoneless } from '../../../../testing/settle-resource';

/**
 * Unit spec for the Auto-Find settings sub-panel. The panel loads the detector
 * registry (an editable Auto-Find checklist) and the pickable exporters (a tab
 * strip whose active tab is the auto-export target), keeping per-exporter field
 * values that it emits to the parent. Both API services are stubbed so the init
 * subscriptions resolve synchronously.
 */
describe('AutoFindSettingsComponent', () => {
  let fixture: ComponentFixture<AutoFindSettingsComponent>;
  let component: AutoFindSettingsComponent;

  const registryDetectors = [
    { id: 'det-1', name: 'Barks', autofind: true, media_type: 'audio' },
    { id: 'det-2', name: 'Cats', autofind: false, media_type: 'image' },
  ];

  const exporters = [
    {
      name: 'server_json_file',
      display_name: 'JSON File',
      description: 'Write results to a JSON file',
      icon: 'download',
      hidden_from_picker: false,
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
      ui_mode: 'form',
      fields: [] as ImporterField[],
    },
    {
      name: 'secret_internal',
      display_name: 'Internal',
      description: 'hidden',
      icon: '',
      hidden_from_picker: true,
      ui_mode: 'form',
      fields: [] as ImporterField[],
    },
  ] as unknown as ExporterEntry[];

  let registryResponse: () => Observable<unknown>;
  let exportersResponse: () => Observable<unknown>;
  let autofindResponse: () => Observable<unknown>;
  let autofindCalls: Array<{ id: string; autofind: boolean }>;

  beforeEach(() => {
    autofindCalls = [];
    registryResponse = () => of({ detectors: registryDetectors });
    exportersResponse = () => of(exporters);
    autofindResponse = () => of({});

    const detectorsStub: Partial<DetectorsRegistryApiService> = {
      getRegistry: () => registryResponse() as Observable<never>,
      setAutofind: (id: string, autofind: boolean) => {
        autofindCalls.push({ id, autofind });
        return autofindResponse() as Observable<never>;
      },
    };
    const exportersStub: Partial<ExportersApiService> = {
      getExporters: () => exportersResponse() as Observable<never>,
    };

    TestBed.configureTestingModule({
      imports: [AutoFindSettingsComponent],
      providers: [
        ...provideZoneless(),
        { provide: DetectorsRegistryApiService, useValue: detectorsStub },
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

  it('creates and loads the detector checklist mapped to the narrowed shape', async () => {
    await create();
    expect(component).toBeTruthy();
    expect(component.loadingDetectors).toBe(false);
    expect(component.detectors).toEqual([
      { id: 'det-1', name: 'Barks', autofind: true, media_type: 'audio' },
      { id: 'det-2', name: 'Cats', autofind: false, media_type: 'image' },
    ]);
    expect(component.detectorError).toBe('');
  });

  it('loads exporters, filtering out picker-hidden ones', async () => {
    await create();
    expect(component.loadingExporters).toBe(false);
    expect(component.exporters.map((e) => e.name)).toEqual(['server_json_file', 'email']);
  });

  it('sets detectorError and clears loading when the registry fails', async () => {
    registryResponse = () => throwError(() => new Error('nope'));
    await create();
    expect(component.detectorError).toBe('Failed to load detectors');
    expect(component.loadingDetectors).toBe(false);
    expect(component.detectors).toEqual([]);
  });

  it('clears the exporter loading flag when the exporters call fails', async () => {
    exportersResponse = () => throwError(() => new Error('nope'));
    await create();
    expect(component.loadingExporters).toBe(false);
    expect(component.exporters).toEqual([]);
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

  it('reacts to input changes via ngOnChanges', async () => {
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

  it('toggleDetector flips the flag optimistically and persists it', async () => {
    await create();
    const cats = component.detectors[1];
    component.toggleDetector(cats, true);
    expect(cats.autofind).toBe(true);
    expect(autofindCalls).toEqual([{ id: 'det-2', autofind: true }]);
  });

  it('toggleDetector reverts and reports an error when the persist fails', async () => {
    autofindResponse = () => throwError(() => new Error('fail'));
    await create();
    const cats = component.detectors[1];
    component.toggleDetector(cats, true);
    expect(cats.autofind).toBe(false); // reverted
    expect(component.detectorError).toBe('Failed to update Auto-Find for "Cats"');
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
    expect(component.inputType({ field_type: 'anything-else' } as ImporterField)).toBe('text');
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
