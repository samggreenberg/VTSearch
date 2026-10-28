import { originParamRows } from './origin-params';

describe('originParamRows', () => {
  it('labels each key the way the hardcoded stat labels read', () => {
    expect(originParamRows({ media_type: 'audio', name: 'esc50' })).toEqual([
      { key: 'media_type', label: 'Media type', value: 'audio' },
      { key: 'name', label: 'Name', value: 'esc50' },
    ]);
  });

  it('drops empty values, and has no rows for no params', () => {
    expect(originParamRows({ path: '', filepath: '/data/labels.json', other: null })).toEqual([
      { key: 'filepath', label: 'Filepath', value: '/data/labels.json' },
    ]);
    expect(originParamRows(undefined)).toEqual([]);
    expect(originParamRows(null)).toEqual([]);
  });
});
