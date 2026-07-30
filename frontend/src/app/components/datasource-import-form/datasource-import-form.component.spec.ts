import { ComponentFixture, TestBed } from '@angular/core/testing';

import { HttpTestingController } from '@angular/common/http/testing';
import { DatasourceImportFormComponent } from './datasource-import-form.component';
import { provideZoneless } from '../../testing/zoneless-testbed';
import { provideHttpTesting } from '../../testing/test-providers';

describe('DatasourceImportFormComponent', () => {
  let component: DatasourceImportFormComponent;
  let fixture: ComponentFixture<DatasourceImportFormComponent>;
  let httpMock: HttpTestingController;

  function setImporter(importer: unknown): void {
    fixture.componentRef.setInput('importer', importer);
    TestBed.tick(); // run the reset effect under zoneless
  }

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [DatasourceImportFormComponent],
      providers: [...provideZoneless(), ...provideHttpTesting()],
    }).compileComponents();

    fixture = TestBed.createComponent(DatasourceImportFormComponent);
    component = fixture.componentInstance;
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => {
    httpMock.verify();
  });

  it('seeds defaults and the first static select option', () => {
    setImporter({
      name: 'imp',
      fields: [
        { key: 'mode', field_type: 'select', options: ['a', 'b'] },
        { key: 'path', field_type: 'text', default: '/tmp/x' },
        { key: 'free', field_type: 'text' },
      ],
    });

    expect(component.values['mode']).toBe('a');
    expect(component.values['path']).toBe('/tmp/x');
    expect(component.values['free']).toBeUndefined();
  });

  it('fetches dynamic options on selection and auto-picks for a required strict select', () => {
    const field = { key: 'sheet', field_type: 'select', dynamic_options: true, required: true };
    setImporter({ name: 'svc', fields: [field] });

    const req = httpMock.expectOne('/api/datasource-import/svc/options');
    expect(req.request.method).toBe('POST');
    expect(req.request.body.field_key).toBe('sheet');
    req.flush({ options: [{ value: 's1', label: 'Sheet 1' }] });

    expect(component.optionsFor(field as never)).toEqual([{ value: 's1', label: 'Sheet 1' }]);
    expect(component.values['sheet']).toBe('s1');
  });

  it('re-fetches a dependent field when its dependency changes', () => {
    const parent = { key: 'doc', field_type: 'select', dynamic_options: true };
    const child = { key: 'tab', field_type: 'select', dynamic_options: true, depends_on: ['doc'] };
    setImporter({ name: 'svc', fields: [parent, child] });
    httpMock.match('/api/datasource-import/svc/options').forEach((r) => r.flush({ options: [] }));

    component.values['doc'] = 'doc-1';
    component.onFieldChanged('doc');

    expect(component.values['tab']).toBe('');
    const req = httpMock.expectOne('/api/datasource-import/svc/options');
    expect(req.request.body.field_key).toBe('tab');
    req.flush({ options: [] });
  });

  it('surfaces a dynamic-option fetch error inline', () => {
    const field = { key: 'q', field_type: 'select', dynamic_options: true };
    setImporter({ name: 'svc', fields: [field] });
    httpMock
      .expectOne('/api/datasource-import/svc/options')
      .flush({ message: 'boom' }, { status: 502, statusText: 'Bad Gateway' });

    expect(component.dynamicError()['q']).toBe('boom');
    expect(component.optionsFor(field as never)).toEqual([]);
  });

  it('blocks submit until every required field is filled', () => {
    setImporter({
      name: 'url_download',
      fields: [{ key: 'url', field_type: 'url', required: true }],
    });
    expect(component.canSubmit).toBe(false);

    component.values['url'] = 'https://example.com/a.wav';
    expect(component.canSubmit).toBe(true);
  });

  it('posts the values and emits the fetched item on success', () => {
    vi.spyOn(component.imported, 'emit');
    setImporter({
      name: 'url_download',
      fields: [{ key: 'url', field_type: 'url', required: true }],
    });
    component.values['url'] = 'https://example.com/a.wav';

    component.submit();

    const req = httpMock.expectOne('/api/datasource-import/url_download');
    expect(req.request.method).toBe('POST');
    expect(req.request.body).toEqual({ url: 'https://example.com/a.wav' });
    req.flush({ filename: 'abc.wav', original_name: 'a.wav' });

    expect(component.imported.emit).toHaveBeenCalledWith({
      filename: 'abc.wav',
      original_name: 'a.wav',
    });
    expect(component.submitting()).toBe(false);
  });

  it('sends multipart when the importer declares a file field', () => {
    setImporter({
      name: 'uploader',
      fields: [{ key: 'file', field_type: 'file', required: true }],
    });
    const file = new File([new Uint8Array([1, 2, 3])], 'clip.wav');
    component.selectedFile = file;
    component.fileFieldKey = 'file';
    component.values['file'] = 'clip.wav';

    component.submit();

    const req = httpMock.expectOne('/api/datasource-import/uploader');
    expect(req.request.body instanceof FormData).toBe(true);
    // FormData.append clones the File under jsdom, so compare by name.
    expect(((req.request.body as FormData).get('file') as File).name).toBe('clip.wav');
    req.flush({ filename: 'x.wav', original_name: 'clip.wav' });
  });

  it('surfaces a run failure without emitting', () => {
    vi.spyOn(component.imported, 'emit');
    setImporter({
      name: 'server_file',
      fields: [{ key: 'path', field_type: 'server_path', required: true }],
    });
    component.values['path'] = '/nope.wav';

    component.submit();
    httpMock
      .expectOne('/api/datasource-import/server_file')
      .flush({ message: 'File not found: /nope.wav' }, { status: 400, statusText: 'Bad Request' });

    expect(component.error()).toBe('File not found: /nope.wav');
    expect(component.imported.emit).not.toHaveBeenCalled();
    expect(component.submitting()).toBe(false);
  });

  it('resets state when the parent selects a different importer', () => {
    setImporter({ name: 'a', fields: [{ key: 'x', field_type: 'text' }] });
    component.values['x'] = 'typed';
    component.error.set('stale');

    setImporter({ name: 'b', fields: [{ key: 'y', field_type: 'text', default: 'd' }] });

    expect(component.values).toEqual({ y: 'd' });
    expect(component.error()).toBe('');
  });
});
