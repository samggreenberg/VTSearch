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

  function renderPrompt(inputs: Record<string, unknown>): HTMLElement {
    for (const [name, value] of Object.entries(inputs)) {
      fixture.componentRef.setInput(name, value);
    }
    fixture.detectChanges();
    return fixture.nativeElement as HTMLElement;
  }

  function squash(text: string | null | undefined): string {
    return (text ?? '').replace(/\s+/g, ' ').trim();
  }

  it('says how the current sort has gone and offers both ways forward', () => {
    const el = renderPrompt({
      currentExampleType: 'text',
      currentExampleDisplay: 'dog barking',
      clicksSoFar: 10,
      positivesSoFar: 1,
      keepLabelsCount: 15,
    });

    expect(squash(el.querySelector('.prompt-text')?.textContent)).toBe(
      'You\'ve clicked 10 times and only found 1 positive while sorting based on ' +
        '\u201cdog barking\u201d. You could keep clicking, or supply a different sort. ' +
        'Which would you like to do?',
    );
    expect(squash(el.querySelector('.keep-btn')?.textContent)).toBe(
      'Keep clicking \u201cdog barking\u201d for 15 more labels.',
    );
    expect(squash(el.querySelector('.choice-col--alt .choice-heading')?.textContent)).toBe(
      'Supply a different sort',
    );
    // Both ways to give a different sort sit under that heading.
    const alt = el.querySelector('.choice-col--alt')!;
    expect(alt.querySelector('input.form-input')).not.toBeNull();
    expect(alt.querySelectorAll('.media-btn').length).toBe(2);
  });

  it('names a media example by its filename and pluralises the counts', () => {
    const el = renderPrompt({
      currentExampleType: 'media',
      currentExampleDisplay: 'bark.wav',
      clicksSoFar: 1,
      positivesSoFar: 0,
      keepLabelsCount: 0,
    });

    expect(squash(el.querySelector('.prompt-text')?.textContent)).toContain(
      'You\'ve clicked 1 time and only found 0 positives while sorting based on bark.wav.',
    );
    // No next-prompt count to promise: the keep option drops the clause.
    expect(squash(el.querySelector('.keep-btn')?.textContent)).toBe('Keep clicking bark.wav.');
  });

  it('keeps the current sort from the left-hand option', () => {
    vi.spyOn(component.keepExample, 'emit');
    const el = renderPrompt({ currentExampleDisplay: 'dog barking' });

    (el.querySelector('.keep-btn') as HTMLButtonElement).click();

    expect(component.keepExample.emit).toHaveBeenCalled();
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
