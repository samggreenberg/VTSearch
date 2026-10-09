import { ComponentFixture, TestBed } from '@angular/core/testing';

import { HttpTestingController } from '@angular/common/http/testing';
import { ServerFolderPickerComponent } from './server-folder-picker.component';
import { provideZoneless } from '../../../../../testing/zoneless-testbed';
import { provideHttpTesting } from '../../../../../testing/test-providers';
import { settleZoneless } from '../../../../../testing/settle-resource';

describe('ServerFolderPickerComponent', () => {
  let component: ServerFolderPickerComponent;
  let fixture: ComponentFixture<ServerFolderPickerComponent>;
  let httpMock: HttpTestingController;

  const serverFolderImporter = {
    name: 'server_folder',
    picker_view: 'server_folder',
    supports_multi_output: true,
    available_converters_by_media_type: {
      image: [{ name: 'document2image', source_type: 'document', target_type: 'image', fields: [] }],
    },
    fields: [{ key: 'media_type', field_type: 'select', default: 'audio', options: ['audio', 'image'] }],
  } as any;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [ServerFolderPickerComponent],
      providers: [...provideZoneless(), ...provideHttpTesting()],
    }).compileComponents();

    fixture = TestBed.createComponent(ServerFolderPickerComponent);
    component = fixture.componentInstance;
    fixture.componentRef.setInput('importers', [serverFolderImporter]);
    fixture.componentRef.setInput('mediaTypes', [
      { type_id: 'audio', name: 'Audio', folder_import_name: 'audio' } as any,
      { type_id: 'image', name: 'Image', folder_import_name: 'image' } as any,
      { type_id: 'document', name: 'Document', folder_import_name: 'document', embeddable: false, converts_to: ['image'] } as any,
    ]);
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => httpMock.verify());

  function openAndFlush(): void {
    component.open(serverFolderImporter);
    httpMock.expectOne(req => req.url === '/api/embedders').flush({ embedders: [] });
    httpMock.expectOne(req => req.url === '/api/clippers').flush({ clippers: [] });
    httpMock.expectOne(req => req.url === '/api/cleaners').flush({ cleaners: [] });
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
    httpMock.expectOne(req => req.url === '/api/cleaners' && req.params.get('media_type') === 'image').flush({ cleaners: [] });
  });

  // ------------------------------------------------------------------
  // Browse (#4207): the browser opens at the typed path and does not
  // overwrite it. These drive the real <vt-folder-browser> over HTTP.
  // ------------------------------------------------------------------

  const isBrowse = (path: string) => (r: { url: string; params: { get(k: string): string | null } }) =>
    r.url === '/api/browse-media-files' && r.params.get('path') === path;
  const isDetect = (r: { url: string }) => r.url === '/api/dataset/detect-media-type';

  /** Type `path` into the field and commit it, as blur/Enter would. */
  function typePath(path: string): void {
    component.onPathInput(path);
    component.applyPathInput();
    httpMock.expectOne(isDetect).flush({ sample_size: 3, counts_by_type: { audio: 3 }, extensions: {}, dominant: 'audio' });
  }

  async function clickBrowse(): Promise<void> {
    const btn = Array.from(fixture.nativeElement.querySelectorAll('button') as NodeListOf<HTMLButtonElement>).find(
      (b) => b.textContent?.trim() === 'Browse',
    );
    btn!.click();
    await settleZoneless(fixture);
  }

  function crumbs(): string[] {
    return Array.from(fixture.nativeElement.querySelectorAll('.vfb-crumb') as NodeListOf<HTMLElement>).map(
      (el) => el.textContent?.trim() ?? '',
    );
  }

  it('Browse opens at the typed path and leaves the field and detection alone', async () => {
    openAndFlush();
    fixture.detectChanges();
    typePath('/srv/photos');

    await clickBrowse();
    // The root is listed first only to learn root_path, then the typed folder.
    httpMock.expectOne(isBrowse('')).flush({ directories: [{ name: 'srv', path: 'srv' }], files: [], root_path: '/' });
    httpMock
      .expectOne(isBrowse('srv/photos'))
      .flush({ directories: [{ name: '2024', path: 'srv/photos/2024' }], files: [], root_path: '/' });
    await settleZoneless(fixture);

    expect(crumbs()).toEqual(['Server root', 'srv', 'photos']);
    expect(component.pathInputValue()).toBe('/srv/photos');
    expect(component.folderPath()).toBe('/srv/photos');
    // Opening the browser is not a pick: no second detection run.
    httpMock.expectNone(isDetect);
  });

  it('Browse keeps a typed path that does not exist, showing the root instead', async () => {
    openAndFlush();
    fixture.detectChanges();
    typePath('/nope');

    await clickBrowse();
    httpMock.expectOne(isBrowse('')).flush({ directories: [{ name: 'srv', path: 'srv' }], files: [], root_path: '/' });
    httpMock.expectOne(isBrowse('nope')).flush({ message: 'Directory not found' }, { status: 404, statusText: 'Not Found' });
    await settleZoneless(fixture);

    expect(crumbs()).toEqual(['Server root']);
    expect(fixture.nativeElement.querySelector('.vfb-error')).toBeNull();
    expect(component.pathInputValue()).toBe('/nope');
    httpMock.expectNone(isDetect);
  });

  it('navigating inside the browser still picks that folder', async () => {
    openAndFlush();
    fixture.detectChanges();
    typePath('/srv/photos');

    await clickBrowse();
    httpMock.expectOne(isBrowse('')).flush({ directories: [], files: [], root_path: '/' });
    httpMock
      .expectOne(isBrowse('srv/photos'))
      .flush({ directories: [{ name: '2024', path: 'srv/photos/2024' }], files: [], root_path: '/' });
    await settleZoneless(fixture);

    const row = fixture.nativeElement.querySelector('.vfb-row--dir') as HTMLElement;
    row.dispatchEvent(new MouseEvent('dblclick'));
    httpMock.expectOne(isBrowse('srv/photos/2024')).flush({ directories: [], files: [], root_path: '/' });
    await settleZoneless(fixture);

    expect(component.pathInputValue()).toBe('/srv/photos/2024');
    expect(crumbs()).toEqual(['Server root', 'srv', 'photos', '2024']);
    httpMock.expectOne(isDetect).flush({ sample_size: 0, counts_by_type: {}, extensions: {}, dominant: null });
  });

  it('with nothing typed, Browse opens at the root and fills the field with it', async () => {
    openAndFlush();
    fixture.detectChanges();

    await clickBrowse();
    httpMock.expectOne(isBrowse('')).flush({ directories: [], files: [], root_path: '/data/alice' });
    await settleZoneless(fixture);

    expect(crumbs()).toEqual(['Server root']);
    expect(component.pathInputValue()).toBe('/data/alice');
    httpMock.expectOne(isDetect).flush({ sample_size: 0, counts_by_type: {}, extensions: {}, dominant: null });
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

  describe('Multi-Dataset mode (#4703)', () => {
    /** The editor loads each ticked type's options; answer them. */
    function flushTypeOptions(mediaType: string): void {
      httpMock.expectOne((r) => r.url === '/api/embedders' && r.params.get('media_type') === mediaType).flush({ embedders: [] });
      httpMock.expectOne((r) => r.url === '/api/clippers' && r.params.get('media_type') === mediaType).flush({ clippers: [] });
      httpMock.expectOne((r) => r.url === '/api/cleaners' && r.params.get('media_type') === mediaType).flush({ cleaners: [] });
    }

    it('is offered when the folder importer supports it, and starts from the dropdown type', async () => {
      openAndFlush();
      expect(component.supportsMultiOutput).toBe(true);
      expect(component.multiDataset()).toBe(false);

      component.setMultiDataset(true);
      await settleZoneless(fixture);
      flushTypeOptions('audio');

      expect(component.multiDataset()).toBe(true);
      expect(component.outputDrafts().map((d) => [d.category, d.checked])).toEqual([
        ['audio', true],
        ['image', false],
        ['document', false],
      ]);
      expect(component.hasOutputs).toBe(true);
      expect(fixture.nativeElement.querySelector('vt-multi-output-config')).not.toBeNull();
      // The single-dataset block is gone; the one Advanced block left is the ticked row's own.
      expect(component.importAdvanced()).toBeUndefined();
      expect(fixture.nativeElement.querySelectorAll('vt-multi-output-config vt-import-advanced').length).toBe(1);
    });

    it('ticks the categories a folder scan finds', async () => {
      openAndFlush();
      component.setMultiDataset(true);
      await settleZoneless(fixture);
      flushTypeOptions('audio');

      component.onPathInput('/data/mixed');
      component.applyPathInput();
      httpMock
        .expectOne((r) => r.url === '/api/dataset/detect-media-type')
        .flush({ sample_size: 4, counts_by_type: { image: 3, document: 1 }, extensions: {}, dominant: 'image' });
      await settleZoneless(fixture);
      // The single-dataset dropdown still follows the dominant type, and its
      // option lists reload for it.
      httpMock.match((r) => r.url === '/api/embedders').forEach((r) => r.flush({ embedders: [] }));
      httpMock.match((r) => r.url === '/api/clippers').forEach((r) => r.flush({ clippers: [] }));
      httpMock.match((r) => r.url === '/api/cleaners').forEach((r) => r.flush({ cleaners: [] }));

      const ticked = component.outputDrafts().filter((d) => d.checked).map((d) => d.category);
      expect(ticked).toEqual(['audio', 'image', 'document']);
    });

    it('submits one outputs entry per ticked row and none of the single-dataset settings', async () => {
      openAndFlush();
      component.onPathInput('/data/mixed');
      component.folderPath.set('/data/mixed');
      component.setMultiDataset(true);
      await settleZoneless(fixture);
      flushTypeOptions('audio');
      component.onOutputDraftsChange(
        component.outputDrafts().map((d) => (d.category === 'document' ? { ...d, checked: true, embedder: 'siglip' } : d)),
      );
      await settleZoneless(fixture);
      flushTypeOptions('image');

      component.submit();
      const req = httpMock.expectOne('/api/dataset/import/server_folder');
      const body = req.request.body as Record<string, unknown>;
      expect(body['path']).toBe('/data/mixed');
      expect(body['media_type']).toBe('audio');
      expect(body['outputs']).toEqual([
        { media_type: 'audio', category: 'audio', source_specs: [{ source_type: 'audio', converter: null, params: {} }] },
        {
          media_type: 'image',
          category: 'document',
          source_specs: [{ source_type: 'document', converter: 'document2image', params: {} }],
          embedder: 'siglip',
        },
      ]);
      expect(body['embedder']).toBeUndefined();
      expect(body['source_specs']).toBeUndefined();
      req.flush({});
    });

    it('has nothing to import with every row unticked', async () => {
      openAndFlush();
      component.setMultiDataset(true);
      await settleZoneless(fixture);
      flushTypeOptions('audio');
      component.onOutputDraftsChange(component.outputDrafts().map((d) => ({ ...d, checked: false })));
      expect(component.hasOutputs).toBe(false);
    });

    it('open() leaves multi mode', async () => {
      openAndFlush();
      component.setMultiDataset(true);
      await settleZoneless(fixture);
      flushTypeOptions('audio');
      openAndFlush();
      expect(component.multiDataset()).toBe(false);
      expect(component.outputDrafts()).toEqual([]);
    });
  });
});
