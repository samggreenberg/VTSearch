import { ComponentFixture, TestBed } from '@angular/core/testing';

import { HttpTestingController } from '@angular/common/http/testing';
import { GenericFormPickerComponent } from './generic-form-picker.component';
import { provideZoneless } from '../../../../../testing/zoneless-testbed';
import { provideHttpTesting } from '../../../../../testing/test-providers';

describe('GenericFormPickerComponent', () => {
  let component: GenericFormPickerComponent;
  let fixture: ComponentFixture<GenericFormPickerComponent>;
  let httpMock: HttpTestingController;

  const importer = {
    name: 'generic_form',
    display_name: 'Generic Form Importer',
    picker_view: 'form',
    fields: [
      { key: 'media_type', field_type: 'select', label: 'Media Type', default: 'audio', options: ['audio', 'image'] },
      { key: 'path', field_type: 'text', label: 'Path', required: true },
    ],
  } as any;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [GenericFormPickerComponent],
      providers: [...provideZoneless(), ...provideHttpTesting()],
    }).compileComponents();

    fixture = TestBed.createComponent(GenericFormPickerComponent);
    component = fixture.componentInstance;
    fixture.componentRef.setInput('importers', [importer]);
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => httpMock.verify());

  function openAndFlush(): void {
    component.open(importer);
    httpMock.expectOne(req => req.url === '/api/embedders').flush({ embedders: [] });
    httpMock.expectOne(req => req.url === '/api/clippers').flush({ clippers: [] });
    httpMock.expectOne(req => req.url === '/api/cleaners').flush({ cleaners: [] });
  }

  it('pre-populates field defaults and requires the required field before submit', () => {
    openAndFlush();
    expect(component.formValues['media_type']).toBe('audio');
    expect(component.canSubmit).toBe(false);
    component.formValues['path'] = '/data/sounds';
    expect(component.canSubmit).toBe(true);
  });

  it('submits via runImporter and reports success through importStarted', () => {
    openAndFlush();
    component.formValues['path'] = '/data/sounds';

    let started = false;
    component.importStarted.subscribe(() => (started = true));
    component.submit();

    const req = httpMock.expectOne('/api/dataset/import/generic_form');
    expect(req.request.body['path']).toBe('/data/sounds');
    req.flush({});

    expect(component.submitting()).toBe(false);
    expect(started).toBe(true);
  });

  it('coerces static string options into {value,label} pairs', () => {
    const staticSelect = { key: 's', field_type: 'select', options: ['x', 'y'] } as any;
    expect(component.fieldOptions.optionsFor(staticSelect)).toEqual([
      { value: 'x', label: 'x' },
      { value: 'y', label: 'y' },
    ]);
  });

  it('does not auto-select the first static option for a free-text combobox', () => {
    const freeTextImporter = {
      name: 'free_text_importer',
      fields: [{ key: 'q', field_type: 'select', options: ['a', 'b'], allow_free_text: true }],
    } as any;
    component.open(freeTextImporter);
    expect(component.formValues['q']).toBeUndefined();
  });

  it('does not auto-select the first option for a required free-text combobox once options load', () => {
    const field = {
      key: 'q',
      field_type: 'select',
      dynamic_options: true,
      required: true,
      allow_free_text: true,
    } as any;
    component.selectedImporter.set({ name: 'imp', fields: [field] } as any);
    component.fieldOptions.refresh(field, component.formValues);
    httpMock
      .expectOne((req) => req.url.endsWith('/api/dataset/import/imp/options'))
      .flush({ options: [{ value: 'a', label: 'A' }] });
    expect(component.formValues['q']).toBeUndefined();
  });

  it('keeps a typed free-text value the refreshed options omit', () => {
    const field = { key: 'q', field_type: 'select', dynamic_options: true, allow_free_text: true } as any;
    component.selectedImporter.set({ name: 'imp', fields: [field] } as any);
    component.formValues['q'] = 'hand-typed';
    component.fieldOptions.refresh(field, component.formValues);
    httpMock
      .expectOne((req) => req.url.endsWith('/api/dataset/import/imp/options'))
      .flush({ options: [{ value: 'a', label: 'A' }] });
    expect(component.formValues['q']).toBe('hand-typed');
  });

  it('clears a strict-select value the refreshed options omit', () => {
    const field = { key: 'q', field_type: 'select', dynamic_options: true } as any;
    component.selectedImporter.set({ name: 'imp', fields: [field] } as any);
    component.formValues['q'] = 'stale';
    component.fieldOptions.refresh(field, component.formValues);
    httpMock
      .expectOne((req) => req.url.endsWith('/api/dataset/import/imp/options'))
      .flush({ options: [{ value: 'a', label: 'A' }] });
    expect(component.formValues['q']).toBe('');
  });

  describe('importer-suggested dataset name', () => {
    // The suggestion round trip is debounced, so every assertion here has to
    // step past SUGGESTED_NAME_DEBOUNCE_MS before the request exists.
    beforeEach(() => vi.useFakeTimers());
    afterEach(() => vi.useRealTimers());

    it('prefills the Dataset Name box with the name the importer would use', () => {
      openAndFlush();
      component.formValues['path'] = '/data/field-recordings';
      component.onFieldChanged('path');
      vi.advanceTimersByTime(300);

      const req = httpMock.expectOne('/api/dataset/import/generic_form/suggested-name');
      expect(req.request.body.values['path']).toBe('/data/field-recordings');
      // The importer is asked what it would pick, not handed the answer.
      expect(req.request.body.values['dataset_name']).toBeUndefined();
      req.flush({ dataset_name: 'Field Recordings' });

      expect(component.formValues['dataset_name']).toBe('Field Recordings');
    });

    it('re-asks once a dynamic select resolves, so an opaque id can become a label', () => {
      const field = { key: 'query_id', field_type: 'select', dynamic_options: true, required: true } as any;
      component.selectedImporter.set({ name: 'imp', fields: [field] } as any);
      component.fieldOptions.refresh(field, component.formValues);
      httpMock
        .expectOne((req) => req.url.endsWith('/api/dataset/import/imp/options'))
        .flush({ options: [{ value: 'q-8f31', label: 'Q1 Field Survey' }] });
      expect(component.formValues['query_id']).toBe('q-8f31');

      vi.advanceTimersByTime(300);
      const req = httpMock.expectOne('/api/dataset/import/imp/suggested-name');
      expect(req.request.body.values['query_id']).toBe('q-8f31');
      req.flush({ dataset_name: 'Q1 Field Survey' });

      expect(component.formValues['dataset_name']).toBe('Q1 Field Survey');
    });

    it('stops asking once the user names the dataset themselves', () => {
      openAndFlush();
      component.onDatasetNameInput('My Corpus');
      component.formValues['path'] = '/data/sounds';
      component.onFieldChanged('path');
      vi.advanceTimersByTime(300);

      httpMock.expectNone((req) => req.url.endsWith('/suggested-name'));
      expect(component.formValues['dataset_name']).toBe('My Corpus');
    });

    it('keeps the current name and stays silent when the importer errors', () => {
      openAndFlush();
      component.formValues['dataset_name'] = 'Earlier Suggestion';
      component.formValues['path'] = '/data/sounds';
      component.onFieldChanged('path');
      vi.advanceTimersByTime(300);

      httpMock
        .expectOne('/api/dataset/import/generic_form/suggested-name')
        .flush({ message: 'upstream down' }, { status: 502, statusText: 'Bad Gateway' });

      expect(component.formValues['dataset_name']).toBe('Earlier Suggestion');
      expect(component.error()).toBe('');
    });

    it('keeps suggesting after a failed request', () => {
      openAndFlush();
      component.formValues['path'] = '/data/one';
      component.onFieldChanged('path');
      vi.advanceTimersByTime(300);
      httpMock
        .expectOne('/api/dataset/import/generic_form/suggested-name')
        .flush({ message: 'upstream down' }, { status: 502, statusText: 'Bad Gateway' });

      component.formValues['path'] = '/data/two';
      component.onFieldChanged('path');
      vi.advanceTimersByTime(300);
      httpMock.expectOne('/api/dataset/import/generic_form/suggested-name').flush({ dataset_name: 'Two' });

      expect(component.formValues['dataset_name']).toBe('Two');
    });
  });

  it('surfaces a server error without emitting importStarted', () => {
    openAndFlush();
    component.formValues['path'] = '/data/sounds';
    component.submit();

    httpMock.expectOne('/api/dataset/import/generic_form').flush(
      { error: 'boom' },
      { status: 500, statusText: 'Internal Server Error' },
    );

    expect(component.submitting()).toBe(false);
    expect(component.error()).toBe('boom');
  });
});
