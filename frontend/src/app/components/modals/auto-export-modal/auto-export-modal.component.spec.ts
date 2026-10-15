import { ComponentFixture, TestBed } from '@angular/core/testing';
import { HttpTestingController } from '@angular/common/http/testing';

import { AutoExportModalComponent } from './auto-export-modal.component';
import { configureZoneless } from '../../../testing/zoneless-testbed';
import { settleZoneless } from '../../../testing/settle-resource';
import { provideHttpTesting } from '../../../testing/test-providers';

/** The JSON source as `GET /api/labelset-sources` lists it. */
const jsonSource = {
  name: 'server_json_file',
  display_name: 'Server JSON File',
  description: '',
  icon: '',
  ui_mode: '',
  hidden_from_picker: false,
  fields: [
    { key: 'filepath', label: 'Save to (server path)', field_type: 'server_path', required: true },
    { key: 'note', label: 'Note', field_type: 'text', required: false },
  ],
};

describe('AutoExportModalComponent', () => {
  let fixture: ComponentFixture<AutoExportModalComponent>;
  let component: AutoExportModalComponent;
  let httpMock: HttpTestingController;

  beforeEach(async () => {
    await configureZoneless({
      imports: [AutoExportModalComponent],
      providers: [...provideHttpTesting()],
    }).compileComponents();
    fixture = TestBed.createComponent(AutoExportModalComponent);
    component = fixture.componentInstance;
    fixture.componentRef.setInput('detectorId', 'det1');
    fixture.componentRef.setInput('detectorName', 'Barks');
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => httpMock.verify());

  async function load(binding: object | null): Promise<void> {
    await fixture.whenStable();
    httpMock.expectOne('/api/detectors/det1/labelset-source').flush(binding);
    httpMock.expectOne('/api/labelset-sources').flush([jsonSource]);
    await settleZoneless(fixture);
  }

  it('asks only for the fields the server did not fix, seeded with the detector’s own values', async () => {
    await load({
      source_name: 'server_json_file',
      display_name: 'Server JSON File',
      field_values: { note: 'mine' },
      inherited: false,
      fixed_keys: ['filepath'],
      missing_fields: [],
    });
    const inputs = fixture.nativeElement.querySelectorAll('vt-labelset-source-fields input');
    expect(inputs.length).toBe(1);
    expect(component.values()).toEqual({ note: 'mine' });
    expect(fixture.nativeElement.textContent).toContain('Server JSON File');
  });

  it('keeps Save disabled until every required field is filled, then PUTs the values', async () => {
    await load({
      source_name: 'server_json_file',
      display_name: 'Server JSON File',
      field_values: {},
      inherited: true,
      fixed_keys: [],
      missing_fields: ['filepath'],
    });
    const save = (): HTMLButtonElement => fixture.nativeElement.querySelector('.btn--primary');
    expect(save().disabled).toBe(true);

    component.values.set({ filepath: '/backups/barks.json', note: '' });
    await settleZoneless(fixture);
    expect(save().disabled).toBe(false);

    let saved = false;
    component.saved.subscribe(() => (saved = true));
    save().click();
    const req = httpMock.expectOne('/api/detectors/det1/labelset-source');
    expect(req.request.method).toBe('PUT');
    expect(req.request.body).toEqual({
      source_name: 'server_json_file',
      field_values: { filepath: '/backups/barks.json', note: '' },
    });
    req.flush({ ok: true, message: 'Labelset source set to Server JSON File.' });
    await settleZoneless(fixture);
    expect(saved).toBe(true);
  });

  it('shows the server’s refusal and stays open', async () => {
    await load({
      source_name: 'server_json_file',
      display_name: 'Server JSON File',
      field_values: { filepath: '../etc' },
      inherited: false,
      fixed_keys: [],
      missing_fields: [],
    });
    component.save();
    httpMock
      .expectOne('/api/detectors/det1/labelset-source')
      .flush({ message: 'Path escapes the allowed directory' }, { status: 400, statusText: 'Bad Request' });
    await settleZoneless(fixture);
    expect(fixture.nativeElement.querySelector('.error-text').textContent).toContain('Path escapes');
  });

  it('says so when the detector syncs nowhere', async () => {
    await load(null);
    expect(fixture.nativeElement.textContent).toContain('not auto-exported anywhere');
    expect(fixture.nativeElement.querySelector('.btn--primary')).toBeFalsy();
  });
});
