import {
  CleanerInfo,
  ClipperInfo,
  ConverterInfo,
  EmbedderInfo,
  ImportDefaultsForMediaType,
  MediaTypeInfo,
} from '../../../../../models/api.models';
import { ImportOptionLists, MAX_LISTED_CHANGES, importDefaultChanges, summarizeChanges } from './import-defaults-summary.util';

// #5535: the "Save as default" offer names each setting saving would change.
describe('importDefaultChanges', () => {
  const mediaTypes: MediaTypeInfo[] = [
    { type_id: 'image', name: 'Image' } as MediaTypeInfo,
    { type_id: 'video', name: 'Video' } as MediaTypeInfo,
    { type_id: 'document', name: 'Document' } as MediaTypeInfo,
    { type_id: 'face', name: 'Face' } as MediaTypeInfo,
  ];
  const embedders = [
    { name: 'siglip', display_name: 'SigLIP 2', is_default: true },
    { name: 'dinov3', display_name: 'DINOv3' },
  ] as EmbedderInfo[];
  const clippers = [
    { name: 'image_default', display_name: 'Whole image', parameters: [] },
    {
      name: 'image_tiling',
      display_name: 'Tiling',
      parameters: [
        { key: 'tiles', label: 'Tiles per side', default: 3 },
        { key: 'overlap', label: 'Overlap', default: 0 },
      ],
    },
  ] as unknown as ClipperInfo[];
  const cleaners = [
    { name: 'image_exif_orient', display_name: 'EXIF Orientation', default_enabled: true, parameters: [] },
    {
      name: 'image_edge_trim',
      display_name: 'Edge Trim',
      default_enabled: false,
      parameters: [{ key: 'max_edge_trim', label: 'Maximum trim per side', default: 0.1 }],
    },
  ] as unknown as CleanerInfo[];
  const converters = [
    {
      name: 'video2image',
      display_name: 'Video → Images',
      source_type: 'video',
      target_type: 'image',
      fields: [{ key: 'n_clips', label: 'Frames per video', default: '8', field_type: 'number' }],
    },
    { name: 'document2image', display_name: 'Document → Images', source_type: 'document', target_type: 'image', fields: [] },
  ] as unknown as ConverterInfo[];
  const lists: ImportOptionLists = { embedders, clippers, cleaners, converters };
  const changes = (saved: ImportDefaultsForMediaType, next: ImportDefaultsForMediaType, l = lists) =>
    importDefaultChanges('image', saved, next, mediaTypes, l);

  it('names nothing when the two configs start an import the same way', () => {
    expect(changes({}, {})).toEqual([]);
    // Naming the built-in embedder outright is the same as leaving it unsaved.
    expect(changes({}, { embedder: 'siglip' })).toEqual([]);
  });

  it('names a picked embedder by its display name, and the built-in one when it goes back', () => {
    expect(changes({}, { embedder: 'dinov3' })).toEqual(['Embedder: DINOv3']);
    expect(changes({ embedder: 'dinov3' }, {})).toEqual(['Embedder: SigLIP 2']);
  });

  it('falls back to registry names when the import offered no lists', () => {
    expect(changes({}, { embedder: 'dinov3', clipper: 'image_tiling' }, {})).toEqual([
      'Embedder: dinov3',
      'Clipper: image_tiling',
    ]);
  });

  it('names a newly picked clipper, and only those of its parameters set away from their defaults', () => {
    expect(changes({}, { clipper: 'image_tiling', clipper_params: { tiles: 3, overlap: 0 } })).toEqual([
      'Clipper: Tiling',
    ]);
    expect(changes({}, { clipper: 'image_tiling', clipper_params: { tiles: 4, overlap: 0 } })).toEqual([
      'Clipper: Tiling',
      'Tiles per side: 4',
    ]);
  });

  it('names a changed parameter of the same clipper, even one set back to its default', () => {
    const saved = { clipper: 'image_tiling', clipper_params: { tiles: 4 } };
    expect(changes(saved, { clipper: 'image_tiling', clipper_params: { tiles: 5 } })).toEqual(['Tiles per side: 5']);
    expect(changes(saved, { clipper: 'image_tiling', clipper_params: {} })).toEqual(['Tiles per side: 3']);
  });

  it('leaves out the parameters of a clipper no longer picked', () => {
    expect(changes({ clipper: 'image_tiling', clipper_params: { tiles: 4 } }, {})).toEqual(['Clipper: Whole image']);
  });

  it('names included media by kind, with a converter parameter set away from its default', () => {
    const video = { source_type: 'video', converter: 'video2image', params: { n_clips: '8' } };
    expect(changes({}, { source_specs: [video] })).toEqual(['Include media: Video']);
    expect(changes({}, { source_specs: [{ ...video, params: { n_clips: '16' } }] })).toEqual([
      'Include media: Video',
      'Frames per video: 16',
    ]);
    expect(changes({ source_specs: [video] }, {})).toEqual(['Include media: none']);
  });

  it('ignores the native row and the order the included kinds are listed in', () => {
    const video = { source_type: 'video', converter: 'video2image', params: {} };
    const doc = { source_type: 'document', converter: 'document2image', params: {} };
    const native = { source_type: 'image', converter: null, params: {} };
    expect(changes({ source_specs: [doc, video] }, { source_specs: [native, video, doc] })).toEqual([]);
    expect(changes({ source_specs: [video] }, { source_specs: [doc, video] })).toEqual([
      'Include media: Video + Document',
    ]);
  });

  it('names each cleaner turned on or off against its default_enabled, and a changed cleaner parameter', () => {
    const cfg = {
      cleaners: { image_exif_orient: { enabled: false }, image_edge_trim: { enabled: true, params: { max_edge_trim: 0.25 } } },
    };
    expect(changes({}, cfg)).toEqual(['EXIF Orientation: off', 'Edge Trim: on', 'Maximum trim per side: 0.25']);
    // Turning a gate back off drops its parameters with it.
    expect(changes(cfg, { cleaners: { image_exif_orient: { enabled: false }, image_edge_trim: { enabled: false } } })).toEqual([
      'Edge Trim: off',
    ]);
  });

  it('names Multi-Dataset by its kinds in registry order, or off', () => {
    expect(changes({}, { multi_dataset: ['face', 'image'] })).toEqual(['Multi-Dataset: Image + Face']);
    expect(changes({ multi_dataset: ['face', 'image'] }, { multi_dataset: ['image', 'face'] })).toEqual([]);
    expect(changes({ multi_dataset: ['image', 'face'] }, {})).toEqual(['Multi-Dataset: off']);
  });

  it('names the two toggles on or off', () => {
    expect(changes({ merge_near_duplicates: true }, { build_projection: true })).toEqual([
      'Build Browse map now: on',
      'Merge near-duplicates: off',
    ]);
  });
});

describe('summarizeChanges', () => {
  it('joins up to the limit with commas', () => {
    expect(summarizeChanges(['Embedder: DINOv3', 'Merge near-duplicates: on'])).toBe(
      'Embedder: DINOv3, Merge near-duplicates: on',
    );
  });

  it('cuts a longer list to the limit and counts the rest', () => {
    const many = Array.from({ length: MAX_LISTED_CHANGES + 2 }, (_, i) => `Setting ${i}: on`);
    expect(summarizeChanges(many)).toBe(`${many.slice(0, MAX_LISTED_CHANGES).join(', ')}, … (2 more)`);
  });
});
