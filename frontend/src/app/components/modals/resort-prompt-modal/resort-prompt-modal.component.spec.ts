import { ComponentFixture, TestBed } from '@angular/core/testing';

import { HttpTestingController } from '@angular/common/http/testing';
import { ResortPromptModalComponent } from './resort-prompt-modal.component';
import { provideZoneless } from '../../../testing/zoneless-testbed';
import { provideHttpTesting } from '../../../testing/test-providers';

describe('ResortPromptModalComponent', () => {
  let component: ResortPromptModalComponent;
  let fixture: ComponentFixture<ResortPromptModalComponent>;
  let httpMock: HttpTestingController;

  /** Open the media picker and answer both source-list requests. */
  function openPicker(
    datasources: unknown[] = [{ name: 'url_download', display_name: 'URL Download', fields: [] }],
  ): void {
    component.openMediaPicker();
    httpMock.expectOne('/api/dataset/all-importers').flush({
      importers: [{ name: 'demo', display_name: 'Demo Datasets' }],
    });
    httpMock.expectOne('/api/datasource-importers').flush({ importers: datasources });
  }

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [ResortPromptModalComponent],
      providers: [...provideZoneless(), ...provideHttpTesting()],
    }).compileComponents();

    fixture = TestBed.createComponent(ResortPromptModalComponent);
    component = fixture.componentInstance;
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => {
    httpMock.verify();
  });

  it('lists datasource importers alongside the browse sources', () => {
    openPicker();

    expect(component.allSources.map((s) => s.name)).toEqual(['demo', 'url_download']);
    expect(component.datasourceImporters().map((s) => s.name)).toEqual(['url_download']);
  });

  it('hides importers flagged out of the picker', () => {
    openPicker([
      { name: 'url_download', fields: [] },
      { name: 'secret', hidden_from_picker: true, fields: [] },
    ]);

    expect(component.datasourceImporters().map((s) => s.name)).toEqual(['url_download']);
  });

  it('renders a datasource importer as a form instead of browsing it', () => {
    openPicker();

    component.selectSource(component.allSources[1]);

    // No browse request goes out; the dynamic form takes over.
    expect(component.selectedDatasourceImporter?.name).toBe('url_download');
    expect(component.browseLoading()).toBe(false);
    expect(component.browseItems()).toEqual([]);
  });

  it('emits the fetched filename as the new media example', () => {
    vi.spyOn(component.newExample, 'emit');
    openPicker();
    component.selectSource(component.allSources[1]);

    component.onDatasourceImported({ filename: 'abc.wav', original_name: 'bark.wav' });

    expect(component.newExample.emit).toHaveBeenCalledWith({
      action: 'new-example',
      type: 'media',
      value: 'abc.wav',
    });
  });

  it('steps back from an importer form to the source list, then to the prompt', () => {
    openPicker();
    component.selectSource(component.allSources[1]);
    expect(component.backLabel).toBe('Back to sources');

    component.back();
    expect(component.selectedSource).toBeNull();
    expect(component.view).toBe('media-picker');
    expect(component.backLabel).toBe('Back');

    component.back();
    expect(component.view).toBe('prompt');
  });

  it('steps back from a demo file entry to the demo list', () => {
    openPicker();
    component.selectSource(component.allSources[0]);
    httpMock.expectOne('/api/dataset/demo-list').flush({
      datasets: [{ name: 'gtzan', label: 'GTZAN', media_type: 'audio', num_files: 10 }],
    });
    component.selectBrowseItem({ key: 'gtzan', display: 'GTZAN' });
    expect(component.fileBrowsing).toBe(true);
    expect(component.backLabel).toBe('Back to demo list');

    component.back();

    expect(component.fileBrowsing).toBe(false);
    expect(component.selectedSource?.name).toBe('demo');
    expect(component.browseItems().length).toBe(1);
  });
});
