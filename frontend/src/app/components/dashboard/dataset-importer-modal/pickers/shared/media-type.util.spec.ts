import { MediaTypeInfo } from '../../../../../models/api.models';
import {
  availableConvertersFor,
  composeEmbedders,
  detectFromFiles,
  detectionHint,
  getTabLabel,
  mediaTypeIconsById,
  mediaTypeLabels,
  mediaTypeOptionIcons,
  mediaTypeOptionLabels,
  readRecursiveDefault,
  toFolderName,
  toTypeId,
} from './media-type.util';

describe('media-type.util', () => {
  const mediaTypes: MediaTypeInfo[] = [
    { type_id: 'audio', name: 'Audio', icon: 'audio', folder_import_name: 'audio', file_extensions: ['.wav', '.mp3'] } as MediaTypeInfo,
    { type_id: 'image', name: 'Image', icon: 'image', folder_import_name: 'images', file_extensions: ['.jpg', '.png'] } as MediaTypeInfo,
  ];

  it('toFolderName maps a type_id to its folder_import_name', () => {
    expect(toFolderName(mediaTypes, 'image')).toBe('images');
    expect(toFolderName(mediaTypes, 'unknown')).toBe('unknown');
    expect(toFolderName(mediaTypes, '')).toBe('');
  });

  it('toTypeId maps a folder_import_name back to its type_id', () => {
    expect(toTypeId(mediaTypes, 'images')).toBe('image');
    expect(toTypeId(mediaTypes, 'unknown')).toBe('unknown');
    expect(toTypeId(mediaTypes, '')).toBe('');
  });

  it('getTabLabel returns the human name, falling back to the raw id', () => {
    expect(getTabLabel(mediaTypes, 'audio')).toBe('Audio');
    expect(getTabLabel(mediaTypes, 'nope')).toBe('nope');
  });

  it('mediaTypeLabels/mediaTypeOptionLabels/mediaTypeOptionIcons build the expected maps', () => {
    expect(mediaTypeLabels(mediaTypes)).toEqual({ audio: 'Audio', image: 'Image' });
    expect(mediaTypeOptionLabels(mediaTypes)).toEqual({ audio: 'Audio', images: 'Image' });
    expect(mediaTypeOptionIcons(mediaTypes)).toEqual({ audio: 'audio', images: 'image' });
    expect(mediaTypeIconsById(mediaTypes)).toEqual({ audio: 'audio', image: 'image' });
  });

  it('detectFromFiles counts extensions and picks the dominant type', () => {
    const files = [
      new File(['a'], 'a.wav'),
      new File(['b'], 'b.wav'),
      new File(['c'], 'c.jpg'),
    ];
    const result = detectFromFiles(mediaTypes, files, true);
    expect(result.dominant).toBe('audio');
    expect(result.counts_by_type['audio']).toBe(2);
    expect(result.counts_by_type['image']).toBe(1);
    expect(result.sample_size).toBe(3);
  });

  it('detectFromFiles skips nested files when recursive is false', () => {
    const nested = new File(['a'], 'nested.wav');
    Object.defineProperty(nested, 'webkitRelativePath', { value: 'top/sub/nested.wav' });
    const topLevel = new File(['b'], 'top.wav');
    Object.defineProperty(topLevel, 'webkitRelativePath', { value: 'top/top.wav' });
    const result = detectFromFiles(mediaTypes, [nested, topLevel], false);
    expect(result.sample_size).toBe(1);
    expect(result.counts_by_type['audio']).toBe(1);
  });

  it('detectionHint describes a single-type detection', () => {
    const hint = detectionHint(mediaTypes, { sample_size: 2, counts_by_type: { audio: 2 }, extensions: {}, dominant: 'audio' });
    expect(hint).toBe('Detected: Audio (2 of 2 files)');
  });

  it('detectionHint describes a mixed-type detection', () => {
    const hint = detectionHint(mediaTypes, {
      sample_size: 3,
      counts_by_type: { audio: 2, image: 1 },
      extensions: {},
      dominant: 'audio',
    });
    expect(hint).toBe('Detected: Audio (2) + Image (1) of 3 files');
  });

  it('detectionHint is empty for a null or empty detection', () => {
    expect(detectionHint(mediaTypes, null)).toBe('');
    expect(detectionHint(mediaTypes, { sample_size: 0, counts_by_type: {}, extensions: {}, dominant: null })).toBe('');
  });

  it('availableConvertersFor reads the importer registry map', () => {
    const importers = [
      {
        name: 'server_folder',
        available_converters_by_media_type: { image: [{ name: 'video_to_image', source_type: 'video', fields: [] }] },
      } as any,
    ];
    expect(availableConvertersFor(importers, 'server_folder', 'image')).toEqual([
      { name: 'video_to_image', source_type: 'video', fields: [] },
    ]);
    expect(availableConvertersFor(importers, 'server_folder', 'audio')).toEqual([]);
    expect(availableConvertersFor(importers, 'missing', 'image')).toEqual([]);
  });

  it('readRecursiveDefault reads the importer field default, falling back to true', () => {
    expect(readRecursiveDefault(null)).toBe(true);
    expect(readRecursiveDefault({ fields: [] } as any)).toBe(true);
    expect(readRecursiveDefault({ fields: [{ key: 'recursive', default: 'false' }] } as any)).toBe(false);
    expect(readRecursiveDefault({ fields: [{ key: 'recursive', default: 'true' }] } as any)).toBe(true);
  });

  it('composeEmbedders dedupes and returns null for a solo primary', () => {
    expect(composeEmbedders('clip', '', '')).toBeNull();
    expect(composeEmbedders('clip', 'clip', '')).toBeNull();
    expect(composeEmbedders('clip', 'patch-clip', '')).toEqual(['clip', 'patch-clip']);
    expect(composeEmbedders('clip', 'patch-clip', 'struct-clip')).toEqual(['clip', 'patch-clip', 'struct-clip']);
  });
});
