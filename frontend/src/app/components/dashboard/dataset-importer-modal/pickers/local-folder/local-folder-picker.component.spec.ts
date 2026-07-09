import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { LocalFolderPickerComponent } from './local-folder-picker.component';
import { provideZoneless } from '../../../../../testing/zoneless-testbed';

describe('LocalFolderPickerComponent', () => {
  let component: LocalFolderPickerComponent;
  let fixture: ComponentFixture<LocalFolderPickerComponent>;
  let httpMock: HttpTestingController;

  const localFolderImporter = { name: 'local_folder', picker_view: 'local_folder', fields: [] } as any;
  const localFilesImporter = { name: 'local_files', picker_view: 'local_files', fields: [] } as any;
  const serverFolderImporter = {
    name: 'server_folder',
    fields: [{ key: 'media_type', field_type: 'select', default: 'audio', options: ['audio', 'image'] }],
  } as any;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [LocalFolderPickerComponent],
      providers: [...provideZoneless(), provideHttpClient(), provideHttpClientTesting()],
    }).compileComponents();

    fixture = TestBed.createComponent(LocalFolderPickerComponent);
    component = fixture.componentInstance;
    component.importers = [localFolderImporter, localFilesImporter, serverFolderImporter];
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => httpMock.verify());

  function openAndFlush(importer = localFolderImporter): void {
    component.open(importer);
    httpMock.expectOne(req => req.url === '/api/embedders').flush({ embedders: [] });
    httpMock.expectOne(req => req.url === '/api/clippers').flush({ clippers: [] });
  }

  it('open() derives pickerKind from the importer name', () => {
    openAndFlush(localFilesImporter);
    expect(component.pickerKind()).toBe('files');
  });

  it('rejects submit with no files selected', () => {
    openAndFlush();
    component.submit();
    expect(component.error()).toContain('folder');
  });

  it('uploads a dropped folder via importLocalFolder and reports success', () => {
    openAndFlush();
    const file = new File(['a'], 'a.wav');
    Object.defineProperty(file, 'webkitRelativePath', { value: 'mydir/a.wav' });
    component.onFilesDropped([file]);
    component.mediaType = 'audio';

    let started = false;
    component.importStarted.subscribe(() => (started = true));
    component.submit();

    const req = httpMock.expectOne('/api/dataset/import-local-folder');
    expect(req.request.method).toBe('POST');
    req.flush({ ok: true });

    expect(component.submitting()).toBe(false);
    expect(started).toBe(true);
  });
});
