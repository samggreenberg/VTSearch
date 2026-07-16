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
    expect(component.optionsFor(staticSelect)).toEqual([
      { value: 'x', label: 'x' },
      { value: 'y', label: 'y' },
    ]);
  });

  it('keeps a typed free-text value the refreshed options omit', () => {
    const field = { key: 'q', field_type: 'select', dynamic_options: true, allow_free_text: true } as any;
    component.selectedImporter.set({ name: 'imp', fields: [field] } as any);
    component.formValues['q'] = 'hand-typed';
    (component as any).refreshDynamicFieldOptions(field);
    httpMock
      .expectOne((req) => req.url.endsWith('/api/dataset/import/imp/options'))
      .flush({ options: [{ value: 'a', label: 'A' }] });
    expect(component.formValues['q']).toBe('hand-typed');
  });

  it('clears a strict-select value the refreshed options omit', () => {
    const field = { key: 'q', field_type: 'select', dynamic_options: true } as any;
    component.selectedImporter.set({ name: 'imp', fields: [field] } as any);
    component.formValues['q'] = 'stale';
    (component as any).refreshDynamicFieldOptions(field);
    httpMock
      .expectOne((req) => req.url.endsWith('/api/dataset/import/imp/options'))
      .flush({ options: [{ value: 'a', label: 'A' }] });
    expect(component.formValues['q']).toBe('');
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
