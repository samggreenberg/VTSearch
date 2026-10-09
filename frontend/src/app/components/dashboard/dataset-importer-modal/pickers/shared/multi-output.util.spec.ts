import { ConverterInfo, DetectMediaTypeResponse, MediaTypeInfo } from '../../../../../models/api.models';
import {
  buildOutputDrafts,
  detectedCount,
  draftToOutputRequest,
  isMultiOutputImporter,
  multiOutputCategories,
  outputsFromDrafts,
  sourceRowsFor,
  tickCategory,
  tickDetectedCategories,
} from './multi-output.util';

describe('multi-output.util', () => {
  const mediaTypes: MediaTypeInfo[] = [
    { type_id: 'audio', name: 'Audio', importable: true, embeddable: true, converts_to: [] } as MediaTypeInfo,
    { type_id: 'document', name: 'Document', importable: true, embeddable: false, converts_to: ['image', 'text'] } as MediaTypeInfo,
    { type_id: 'face', name: 'Face', importable: false, embeddable: true, converts_to: [] } as MediaTypeInfo,
    { type_id: 'image', name: 'Image', importable: true, embeddable: true, converts_to: [] } as MediaTypeInfo,
    { type_id: 'hologram', name: 'Hologram', importable: false, embeddable: true, converts_to: [] } as MediaTypeInfo,
  ];
  const convertersByType: Record<string, ConverterInfo[]> = {
    image: [
      { name: 'document2image', source_type: 'document', target_type: 'image', fields: [{ key: 'dpi', field_type: 'number', default: '72' }] } as ConverterInfo,
      { name: 'video2image', source_type: 'video', target_type: 'image', fields: [] } as ConverterInfo,
    ],
    text: [{ name: 'document2text', source_type: 'document', target_type: 'text', fields: [] } as ConverterInfo],
    face: [{ name: 'image2face', source_type: 'image', target_type: 'face', fields: [] } as ConverterInfo],
  };

  it('isMultiOutputImporter reads the importer flag', () => {
    expect(isMultiOutputImporter({ supports_multi_output: true } as any)).toBe(true);
    expect(isMultiOutputImporter({ supports_multi_output: false } as any)).toBe(false);
    expect(isMultiOutputImporter({} as any)).toBe(false);
    expect(isMultiOutputImporter(null)).toBe(false);
  });

  it('lists importable categories plus convert-in types a converter can produce', () => {
    expect(multiOutputCategories(mediaTypes, convertersByType).map((m) => m.type_id)).toEqual([
      'audio',
      'document',
      'face',
      'image',
    ]);
  });

  describe('sourceRowsFor', () => {
    it('an embeddable category keeps its type with a direct row', () => {
      expect(sourceRowsFor(mediaTypes[3], convertersByType)).toEqual({
        mediaType: 'image',
        convertTarget: '',
        sourceSpecs: [{ source_type: 'image', converter: null, params: {} }],
      });
    });

    it('a convert-out category becomes a dataset of its default target', () => {
      expect(sourceRowsFor(mediaTypes[1], convertersByType)).toEqual({
        mediaType: 'image',
        convertTarget: 'image',
        sourceSpecs: [{ source_type: 'document', converter: 'document2image', params: { dpi: '72' } }],
      });
    });

    it('a convert-out category follows the chosen target', () => {
      expect(sourceRowsFor(mediaTypes[1], convertersByType, 'text')).toEqual({
        mediaType: 'text',
        convertTarget: 'text',
        sourceSpecs: [{ source_type: 'document', converter: 'document2text', params: {} }],
      });
    });

    it('a convert-in category is fed by every converter into it, with no native row', () => {
      expect(sourceRowsFor(mediaTypes[2], convertersByType)).toEqual({
        mediaType: 'face',
        convertTarget: '',
        sourceSpecs: [{ source_type: 'image', converter: 'image2face', params: {} }],
      });
    });
  });

  it('buildOutputDrafts seeds one unticked row per category', () => {
    const drafts = buildOutputDrafts(mediaTypes, convertersByType);
    expect(drafts.map((d) => [d.category, d.mediaType, d.checked])).toEqual([
      ['audio', 'audio', false],
      ['document', 'image', false],
      ['face', 'face', false],
      ['image', 'image', false],
    ]);
    expect(drafts.every((d) => d.embedder === '' && d.clipper === '' && d.cleaners.length === 0)).toBe(true);
  });

  it('tickDetectedCategories ticks what the scan found and leaves the rest alone', () => {
    const detection: DetectMediaTypeResponse = {
      sample_size: 5,
      counts_by_type: { image: 3, document: 2 },
      extensions: {},
      dominant: 'image',
      truncated: false,
    };
    const drafts = tickCategory(buildOutputDrafts(mediaTypes, convertersByType), 'audio');
    const ticked = tickDetectedCategories(drafts, detection);
    expect(ticked.filter((d) => d.checked).map((d) => d.category)).toEqual(['audio', 'document', 'image']);
    expect(tickDetectedCategories(drafts, null)).toBe(drafts);
    expect(detectedCount(detection, 'document')).toBe(2);
    expect(detectedCount(detection, 'face')).toBe(0);
    expect(detectedCount(null, 'image')).toBe(0);
  });

  describe('draftToOutputRequest', () => {
    const base = buildOutputDrafts(mediaTypes, convertersByType);

    it('sends only what is set', () => {
      const image = base.find((d) => d.category === 'image')!;
      expect(draftToOutputRequest(image)).toEqual({
        media_type: 'image',
        category: 'image',
        source_specs: [{ source_type: 'image', converter: null, params: {} }],
      });
    });

    it('carries the per-dataset settings and composes the embedder trio', () => {
      const document = base.find((d) => d.category === 'document')!;
      const request = draftToOutputRequest({
        ...document,
        embedder: 'siglip',
        patchEmbedder: 'dinov3',
        structuralEmbedder: '',
        clipper: 'image_default',
        clipperParams: { tiles: 4 },
        cleaners: [{ name: 'exif', params: {} }],
      });
      expect(request).toEqual({
        media_type: 'image',
        category: 'document',
        source_specs: [{ source_type: 'document', converter: 'document2image', params: { dpi: '72' } }],
        embedder: 'siglip',
        embedders: ['siglip', 'dinov3'],
        clipper: 'image_default',
        clipper_params: { tiles: 4 },
        cleaners: [{ name: 'exif', params: {} }],
      });
    });

    it('outputsFromDrafts keeps ticked rows in category order', () => {
      const drafts = tickCategory(tickCategory(base, 'image'), 'audio');
      expect(outputsFromDrafts(drafts).map((o) => o.category)).toEqual(['audio', 'image']);
    });
  });
});
