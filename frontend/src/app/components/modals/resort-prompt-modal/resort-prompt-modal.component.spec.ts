import { ComponentFixture, TestBed } from '@angular/core/testing';

import { HttpTestingController } from '@angular/common/http/testing';
import { ResortPromptModalComponent } from './resort-prompt-modal.component';
import { provideZoneless } from '../../../testing/zoneless-testbed';
import { provideHttpTesting } from '../../../testing/test-providers';
import { settleZoneless } from '../../../testing/settle-resource';

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

  /** The status line's fields, as "Label: value". */
  function statusFields(el: HTMLElement): string[] {
    return [...el.querySelectorAll('.sort-status__field')].map((f) =>
      squash(`${f.querySelector('dt')?.textContent} ${f.querySelector('dd')?.textContent}`),
    );
  }

  it('reports the current sort as read-only fields and offers both ways forward (#4721)', () => {
    const el = renderPrompt({
      currentExampleType: 'text',
      currentExampleDisplay: 'dog barking',
      clicksSoFar: 10,
      positivesSoFar: 1,
      positivesNeeded: 3,
    });

    expect(statusFields(el)).toEqual(['Clicked: 10', 'Positives: 1', 'Sort: \u201cdog barking\u201d']);
    // The fields are the whole of the dialog's own text: the explaining is Toasty's.
    expect(el.querySelector('.prompt-text')).toBeNull();
    expect([...el.querySelectorAll('.choice-heading')].map((h) => squash(h.textContent))).toEqual([
      'Keep clicking:',
      'Supply a different sort:',
    ]);
    // Both ways to give a different sort sit under the right-hand heading, the
    // two media buttons on one line.
    const alt = el.querySelector('.choice-col--alt')!;
    expect(alt.querySelector('input.form-input')).not.toBeNull();
    expect(alt.querySelectorAll('.media-btn-row .media-btn').length).toBe(2);
  });

  it('labels the keep button the same whatever the sort', () => {
    const el = renderPrompt({ currentExampleType: 'media', currentExampleDisplay: 'book.png', clicksSoFar: 35 });

    expect(squash(el.querySelector('.keep-btn')?.textContent)).toBe('Continue');
  });

  it('names a media example by its filename', () => {
    const el = renderPrompt({ currentExampleType: 'media', currentExampleDisplay: 'bark.wav' });

    expect(statusFields(el)[2]).toBe('Sort: bark.wav');
    expect(el.querySelector('.sort-status__field--sort dd')?.getAttribute('title')).toBe('bark.wav');
  });

  it("has Toasty explain the prompt below the dialog box (#4721)", async () => {
    renderPrompt({ positivesNeeded: 3 });
    await settleZoneless(fixture);
    const el = fixture.nativeElement as HTMLElement;

    const hint = el.querySelector('vt-toasty-hint');
    expect(hint).not.toBeNull();
    expect(squash(hint!.querySelector('.toasty-hint__text')?.textContent)).toBe(
      'We need 3 positives before Autopilot can move on. This is an opportunity to try a different ' +
        'example sort, or just keep clicking with the original sort.',
    );

    fixture.componentRef.setInput('positivesNeeded', 1);
    await settleZoneless(fixture);
    expect(squash(hint!.textContent)).toContain('We need 1 positive before Autopilot can move on.');
  });

  it('opens with focus on the dialog box, not on an answer (#4721)', () => {
    const el = renderPrompt({ currentExampleDisplay: 'dog barking' });

    // A ring on the keep button read as the recommended answer, and the next
    // Space or Enter took it.
    const box = el.querySelector('.modal-content')!;
    expect(box.hasAttribute('cdkFocusInitial')).toBe(true);
    expect(box.getAttribute('tabindex')).toBe('-1');
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
