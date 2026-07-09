import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { ServerFolderPickerComponent } from './server-folder-picker.component';
import { provideZoneless } from '../../../../../testing/zoneless-testbed';

describe('ServerFolderPickerComponent', () => {
  let component: ServerFolderPickerComponent;
  let fixture: ComponentFixture<ServerFolderPickerComponent>;
  let httpMock: HttpTestingController;

  const serverFolderImporter = {
    name: 'server_folder',
    picker_view: 'server_folder',
    fields: [{ key: 'media_type', field_type: 'select', default: 'audio', options: ['audio', 'image'] }],
  } as any;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [ServerFolderPickerComponent],
      providers: [...provideZoneless(), provideHttpClient(), provideHttpClientTesting()],
    }).compileComponents();

    fixture = TestBed.createComponent(ServerFolderPickerComponent);
    component = fixture.componentInstance;
    component.importers = [serverFolderImporter];
    component.mediaTypes = [
      { type_id: 'audio', name: 'Audio', folder_import_name: 'audio' } as any,
      { type_id: 'image', name: 'Image', folder_import_name: 'image' } as any,
    ];
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => httpMock.verify());

  function openAndFlush(): void {
    component.open(serverFolderImporter);
    httpMock.expectOne(req => req.url === '/api/embedders').flush({ embedders: [] });
    httpMock.expectOne(req => req.url === '/api/clippers').flush({ clippers: [] });
  }

  it('open() resets the form to the default media type', () => {
    openAndFlush();
    expect(component.mediaType()).toBe('audio');
    expect(component.folderPath()).toBe('');
  });

  it('applyPathInput commits the path, derives a dataset name, and runs detection', () => {
    openAndFlush();
    component.onPathInput('/data/my-photos/');
    component.applyPathInput();

    expect(component.folderPath()).toBe('/data/my-photos');
    expect(component.datasetName).toBe('my-photos');

    const req = httpMock.expectOne(r => r.url === '/api/dataset/detect-media-type');
    req.flush({ sample_size: 3, counts_by_type: { image: 3 }, extensions: {}, dominant: 'image' });

    expect(component.mediaType()).toBe('image');
    httpMock.expectOne(req => req.url === '/api/embedders' && req.params.get('media_type') === 'image').flush({ embedders: [] });
    httpMock.expectOne(req => req.url === '/api/clippers' && req.params.get('media_type') === 'image').flush({ clippers: [] });
  });

  it('submit() posts to runImporter with the current path and media type', () => {
    openAndFlush();
    component.onPathInput('/data/sounds');
    component.folderPath.set('/data/sounds');

    let started = false;
    component.importStarted.subscribe(() => (started = true));
    component.submit();

    const req = httpMock.expectOne('/api/dataset/import/server_folder');
    expect(req.request.body).toEqual(
      expect.objectContaining({ path: '/data/sounds', media_type: 'audio' }),
    );
    req.flush({});

    expect(component.submitting()).toBe(false);
    expect(started).toBe(true);
  });
});
