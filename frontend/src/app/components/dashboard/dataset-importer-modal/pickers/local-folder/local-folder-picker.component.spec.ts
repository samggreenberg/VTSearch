import { ComponentFixture, TestBed } from '@angular/core/testing';

import { HttpTestingController } from '@angular/common/http/testing';
import { LocalFolderPickerComponent } from './local-folder-picker.component';
import { provideZoneless } from '../../../../../testing/zoneless-testbed';
import { provideHttpTesting } from '../../../../../testing/test-providers';
import { settleZoneless } from '../../../../../testing/settle-resource';

describe('LocalFolderPickerComponent', () => {
  let component: LocalFolderPickerComponent;
  let fixture: ComponentFixture<LocalFolderPickerComponent>;
  let httpMock: HttpTestingController;

  const localFolderImporter = { name: 'local_folder', picker_view: 'local_folder', fields: [] } as any;
  const localFilesImporter = { name: 'local_files', picker_view: 'local_files', fields: [] } as any;
  const serverFolderImporter = {
    name: 'server_folder',
    supports_multi_output: true,
    available_converters_by_media_type: {},
    fields: [{ key: 'media_type', field_type: 'select', default: 'audio', options: ['audio', 'image'] }],
  } as any;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [LocalFolderPickerComponent],
      providers: [...provideZoneless(), ...provideHttpTesting()],
    }).compileComponents();

    fixture = TestBed.createComponent(LocalFolderPickerComponent);
    component = fixture.componentInstance;
    fixture.componentRef.setInput('importers', [localFolderImporter, localFilesImporter, serverFolderImporter]);
    fixture.componentRef.setInput('mediaTypes', [
      { type_id: 'audio', name: 'Audio', folder_import_name: 'audio', file_extensions: ['*.wav'] } as any,
      { type_id: 'image', name: 'Image', folder_import_name: 'image', file_extensions: ['*.png'] } as any,
    ]);
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => httpMock.verify());

  function openAndFlush(importer = localFolderImporter): void {
    component.open(importer);
    httpMock.expectOne(req => req.url === '/api/embedders').flush({ embedders: [] });
    httpMock.expectOne(req => req.url === '/api/clippers').flush({ clippers: [] });
    httpMock.expectOne(req => req.url === '/api/cleaners').flush({ cleaners: [] });
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

  describe('Multi-Dataset mode (#4703)', () => {
    function flushTypeOptions(mediaType: string): void {
      httpMock.expectOne((r) => r.url === '/api/embedders' && r.params.get('media_type') === mediaType).flush({ embedders: [] });
      httpMock.expectOne((r) => r.url === '/api/clippers' && r.params.get('media_type') === mediaType).flush({ clippers: [] });
      httpMock.expectOne((r) => r.url === '/api/cleaners' && r.params.get('media_type') === mediaType).flush({ cleaners: [] });
    }

    it('is offered for the folder upload (the server_folder importer supports it)', () => {
      openAndFlush();
      expect(component.supportsMultiOutput).toBe(true);
      openAndFlush(localFilesImporter);
      // No server_files importer in the mocks: nothing says it supports it.
      expect(component.supportsMultiOutput).toBe(false);
    });

    it('ticks what the dropped files were found to be, and uploads one outputs entry per ticked row', async () => {
      openAndFlush();
      const wav = new File(['a'], 'a.wav');
      Object.defineProperty(wav, 'webkitRelativePath', { value: 'mydir/a.wav' });
      const png = new File(['b'], 'b.png');
      Object.defineProperty(png, 'webkitRelativePath', { value: 'mydir/b.png' });
      component.onFilesDropped([wav, png]);
      // The dropped files are mostly audio, so the single-dataset dropdown
      // settled on audio (no reload needed: it was audio already).

      component.setMultiDataset(true);
      await settleZoneless(fixture);
      flushTypeOptions('audio');
      flushTypeOptions('image');
      expect(component.outputDrafts().filter((d) => d.checked).map((d) => d.category)).toEqual(['audio', 'image']);

      component.submit();
      const req = httpMock.expectOne('/api/dataset/import-local-folder');
      const form = req.request.body as FormData;
      const outputs = JSON.parse(form.get('outputs') as string);
      expect(outputs.map((o: { category: string }) => o.category)).toEqual(['audio', 'image']);
      expect(form.get('embedder')).toBeNull();
      expect(form.get('source_specs')).toBeNull();
      expect(form.getAll('files').length).toBe(2);
      req.flush({ ok: true });
      expect(component.submitting()).toBe(false);
    });

    it('refuses to upload with every row unticked', async () => {
      openAndFlush();
      const wav = new File(['a'], 'a.wav');
      Object.defineProperty(wav, 'webkitRelativePath', { value: 'mydir/a.wav' });
      component.onFilesDropped([wav]);
      component.setMultiDataset(true);
      await settleZoneless(fixture);
      flushTypeOptions('audio');
      component.onOutputDraftsChange(component.outputDrafts().map((d) => ({ ...d, checked: false })));
      component.submit();
      expect(component.error()).toContain('Tick at least one');
      httpMock.expectNone('/api/dataset/import-local-folder');
    });
  });
});
