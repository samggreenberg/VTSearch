import { ComponentFixture, TestBed } from '@angular/core/testing';

import { HttpTestingController } from '@angular/common/http/testing';
import { MultiOutputConfigComponent } from './multi-output-config.component';
import { provideZoneless } from '../../../../testing/zoneless-testbed';
import { provideHttpTesting } from '../../../../testing/test-providers';
import { settleZoneless } from '../../../../testing/settle-resource';
import { ConverterInfo, MediaTypeInfo } from '../../../../models/api.models';
import { OutputDraft, buildOutputDrafts, tickCategory } from '../pickers/shared/multi-output.util';

describe('MultiOutputConfigComponent', () => {
  let component: MultiOutputConfigComponent;
  let fixture: ComponentFixture<MultiOutputConfigComponent>;
  let httpMock: HttpTestingController;
  let drafts: OutputDraft[];

  const mediaTypes: MediaTypeInfo[] = [
    { type_id: 'audio', name: 'Audio', icon: 'audio', importable: true, embeddable: true, converts_to: [] } as MediaTypeInfo,
    { type_id: 'document', name: 'Document', icon: 'document', importable: true, embeddable: false, converts_to: ['image', 'text'] } as MediaTypeInfo,
    { type_id: 'face', name: 'Face', icon: 'face', importable: false, embeddable: true, converts_to: [] } as MediaTypeInfo,
    { type_id: 'image', name: 'Image', icon: 'image', importable: true, embeddable: true, converts_to: [] } as MediaTypeInfo,
  ];
  const convertersByType: Record<string, ConverterInfo[]> = {
    image: [{ name: 'document2image', source_type: 'document', target_type: 'image', fields: [] } as ConverterInfo],
    text: [{ name: 'document2text', source_type: 'document', target_type: 'text', fields: [] } as ConverterInfo],
    face: [{ name: 'image2face', source_type: 'image', target_type: 'face', fields: [] } as ConverterInfo],
  };

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [MultiOutputConfigComponent],
      providers: [...provideZoneless(), ...provideHttpTesting()],
    }).compileComponents();

    fixture = TestBed.createComponent(MultiOutputConfigComponent);
    component = fixture.componentInstance;
    httpMock = TestBed.inject(HttpTestingController);
    drafts = buildOutputDrafts(mediaTypes, convertersByType);
    fixture.componentRef.setInput('mediaTypes', mediaTypes);
    fixture.componentRef.setInput('convertersByType', convertersByType);
    fixture.componentRef.setInput('drafts', drafts);
    // Mirror the two-way binding the pickers provide.
    component.draftsChange.subscribe((next) => {
      drafts = next;
      fixture.componentRef.setInput('drafts', next);
    });
    await settleZoneless(fixture);
  });

  afterEach(() => httpMock.verify());

  /** Answer the three option fetches for *mediaType*. */
  function flushOptions(mediaType: string, embedders: Array<Record<string, unknown>> = [{ name: 'e1', is_default: true }]): void {
    httpMock.expectOne((r) => r.url === '/api/embedders' && r.params.get('media_type') === mediaType).flush({ embedders });
    httpMock
      .expectOne((r) => r.url === '/api/clippers' && r.params.get('media_type') === mediaType)
      .flush({ clippers: [{ name: `${mediaType}_default`, parameters: [{ key: 'n', default: 3 }] }] });
    httpMock.expectOne((r) => r.url === '/api/cleaners' && r.params.get('media_type') === mediaType).flush({
      cleaners: [{ name: 'tidy', default_enabled: true }],
    });
  }

  function rows(): HTMLLIElement[] {
    return Array.from(fixture.nativeElement.querySelectorAll('.mo-row'));
  }

  it('renders one row per category, unticked, with no Details link', () => {
    expect(rows().map((r) => r.querySelector('.mo-row-label')?.textContent?.trim())).toEqual(['Audio', 'Document', 'Face', 'Image']);
    expect(fixture.nativeElement.querySelectorAll('.mo-details-toggle').length).toBe(0);
    expect(component.anyChecked).toBe(false);
    expect(fixture.nativeElement.querySelector('.info-text')?.textContent).toContain('Tick at least one');
  });

  it('ticking a row loads its type options once and seeds its defaults', async () => {
    component.toggle(drafts[3], true);
    flushOptions('image');
    await settleZoneless(fixture);

    const image = drafts.find((d) => d.category === 'image')!;
    expect(image.checked).toBe(true);
    expect(image.embedder).toBe('e1');
    expect(image.clipper).toBe('image_default');
    expect(image.clipperParams).toEqual({ n: 3 });
    expect(image.cleaners).toEqual([{ name: 'tidy', params: {} }]);
    expect(fixture.nativeElement.querySelectorAll('.mo-details-toggle').length).toBe(1);
    expect(fixture.nativeElement.querySelectorAll('vt-import-advanced').length).toBe(1);

    // A second row of the same dataset type reuses the loaded options.
    component.toggle(drafts.find((d) => d.category === 'document')!, true);
    await settleZoneless(fixture);
    httpMock.expectNone((r) => r.url === '/api/embedders');
    expect(drafts.find((d) => d.category === 'document')!.embedder).toBe('e1');
  });

  it('rows ticked from outside (a folder scan) load their options too', async () => {
    fixture.componentRef.setInput('drafts', tickCategory(drafts, 'audio'));
    await settleZoneless(fixture);
    flushOptions('audio');
    await settleZoneless(fixture);
    expect(component.typeOptions()['audio'].embedders.map((e) => e.name)).toEqual(['e1']);
  });

  it('the Details link opens the row’s own Advanced block', async () => {
    component.toggle(drafts[3], true);
    flushOptions('image');
    await settleZoneless(fixture);

    const block = fixture.nativeElement.querySelector('vt-import-advanced') as HTMLElement;
    expect(block.classList.contains('is-open')).toBe(false);
    (fixture.nativeElement.querySelector('.mo-details-toggle') as HTMLButtonElement).click();
    await settleZoneless(fixture);
    expect(block.classList.contains('is-open')).toBe(true);
    // The two shared ingest toggles are not repeated inside the row's block.
    expect(block.querySelector('#import-advanced-projection')).toBeNull();
    expect(fixture.nativeElement.querySelector('#mo-projection')).not.toBeNull();
  });

  it('a convert-out row re-targets its dataset type and reseeds from the new type', async () => {
    const document = drafts.find((d) => d.category === 'document')!;
    component.toggle(document, true);
    flushOptions('image');
    await settleZoneless(fixture);

    component.onConvertTargetChange(drafts.find((d) => d.category === 'document')!, 'text');
    flushOptions('text', [{ name: 'e5', is_default: true }]);
    await settleZoneless(fixture);

    const retargeted = drafts.find((d) => d.category === 'document')!;
    expect(retargeted.mediaType).toBe('text');
    expect(retargeted.convertTarget).toBe('text');
    expect(retargeted.sourceSpecs).toEqual([{ source_type: 'document', converter: 'document2text', params: {} }]);
    expect(retargeted.embedder).toBe('e5');
  });

  it('a convert-out row offers Convert to but no include rows; a convert-in row the reverse', () => {
    const document = drafts.find((d) => d.category === 'document')!;
    const face = drafts.find((d) => d.category === 'face')!;
    expect(component.isConvertOut(document)).toBe(true);
    expect(component.convertTargets(document)).toEqual(['image', 'text']);
    expect(component.convertersFor(document)).toEqual([]);
    expect(component.isConvertOut(face)).toBe(false);
    expect(component.nativeImportable(face)).toBe(false);
    expect(component.convertersFor(face).map((c) => c.name)).toEqual(['image2face']);
  });

  it('shows the scan count beside a row and names the row’s meaning in its tooltip', async () => {
    fixture.componentRef.setInput('detection', {
      sample_size: 4,
      counts_by_type: { image: 3, document: 1 },
      extensions: {},
      dominant: 'image',
      truncated: false,
    });
    await settleZoneless(fixture);
    const counts = rows().map((r) => r.querySelector('.mo-row-count')?.textContent?.trim() ?? '');
    expect(counts).toEqual(['', '1 file', '', '3 files']);
    expect(component.rowTitle(drafts[1])).toContain('converted to Image');
    expect(component.rowTitle(drafts[2])).toContain('converters');
  });

  it('a clipper request names the row and its clippers; setClipper lands on that row', async () => {
    component.toggle(drafts[3], true);
    flushOptions('image');
    await settleZoneless(fixture);

    let request: { category: string; clippers: unknown[] } | undefined;
    component.clipperChooserRequested.subscribe((r) => (request = r));
    component.onClipperRequested(drafts.find((d) => d.category === 'image')!);
    expect(request?.category).toBe('image');
    expect(request?.clippers.length).toBe(1);

    component.setClipper('image', 'sliding', { n: 9 });
    const image = drafts.find((d) => d.category === 'image')!;
    expect(image.clipper).toBe('sliding');
    expect(image.clipperParams).toEqual({ n: 9 });
  });

  it('forwards the shared ingest toggles', () => {
    let projection: boolean | undefined;
    component.buildProjectionChange.subscribe((v) => (projection = v));
    (fixture.nativeElement.querySelector('#mo-projection') as HTMLInputElement).click();
    expect(projection).toBe(true);
  });
});
