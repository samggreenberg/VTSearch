import { TestBed } from '@angular/core/testing';
import { ActiveContextService } from './active-context.service';
import { configureZoneless } from '../testing/zoneless-testbed';

/**
 * Specs for the source-of-truth singleton behind the whole context-selection
 * pipeline. The service holds two layers — **intent** (what the user picked)
 * and **active** (what the backend has loaded, which the HTTP interceptor tags
 * onto requests via `X-Dataset-Id` / `X-Detector-Id`). These tests pin the
 * layer independence, the no-op-on-unchanged emission contract, and the
 * request-id / media-url helpers the switcher and native-media paths rely on.
 */
describe('ActiveContextService', () => {
  let svc: ActiveContextService;

  beforeEach(() => {
    configureZoneless();
    svc = TestBed.inject(ActiveContextService);
  });

  it('starts with both layers empty', () => {
    expect(svc.datasetId).toBe('');
    expect(svc.modelId).toBe('');
    expect(svc.intentDatasetId).toBe('');
    expect(svc.intentModelId).toBe('');
    expect(svc.currentRequestId).toBe(0);
  });

  describe('layer independence', () => {
    it('setIntent moves the intent layer but leaves active pinned', () => {
      svc.setIntent('d1', 'm1');
      expect(svc.intentDatasetId).toBe('d1');
      expect(svc.intentModelId).toBe('m1');
      // Active is what the interceptor reads; it must NOT move until a load
      // has finished and the switcher promotes it.
      expect(svc.datasetId).toBe('');
      expect(svc.modelId).toBe('');
    });

    it('setActive moves the active layer but leaves intent untouched', () => {
      svc.setActive('d1', 'm1');
      expect(svc.datasetId).toBe('d1');
      expect(svc.modelId).toBe('m1');
      expect(svc.intentDatasetId).toBe('');
      expect(svc.intentModelId).toBe('');
    });

    it('setActivePair writes both layers atomically', () => {
      svc.setActivePair('d1', 'm1');
      expect(svc.datasetId).toBe('d1');
      expect(svc.modelId).toBe('m1');
      expect(svc.intentDatasetId).toBe('d1');
      expect(svc.intentModelId).toBe('m1');
    });

    it('clear() resets both layers to empty', () => {
      svc.setActivePair('d1', 'm1');
      svc.clear();
      expect(svc.datasetId).toBe('');
      expect(svc.modelId).toBe('');
      expect(svc.intentDatasetId).toBe('');
      expect(svc.intentModelId).toBe('');
    });
  });

  describe('pair$ emissions', () => {
    it('emits once per atomic setActive, not once per half', () => {
      const pairs: { datasetId: string; modelId: string }[] = [];
      svc.pair$.subscribe((p) => pairs.push(p));
      // Initial replay of the empty pair.
      expect(pairs).toEqual([{ datasetId: '', modelId: '' }]);

      svc.setActive('d1', 'm1');
      // One emission for the pair change, not two (one per half).
      expect(pairs.length).toBe(2);
      expect(pairs[1]).toEqual({ datasetId: 'd1', modelId: 'm1' });
    });

    it('does not re-emit when setActive is called with the current pair', () => {
      svc.setActive('d1', 'm1');
      const pairs: { datasetId: string; modelId: string }[] = [];
      svc.pair$.subscribe((p) => pairs.push(p));
      expect(pairs.length).toBe(1); // replayed current value

      svc.setActive('d1', 'm1'); // no-op
      expect(pairs.length).toBe(1);
    });

    it('intentPair$ tracks the intent layer independently of active', () => {
      const pairs: { datasetId: string; modelId: string }[] = [];
      svc.intentPair$.subscribe((p) => pairs.push(p));
      expect(pairs.length).toBe(1);

      svc.setIntent('d1', 'm1');
      expect(pairs[pairs.length - 1]).toEqual({ datasetId: 'd1', modelId: 'm1' });

      // Promoting active alone must NOT push a new intentPair emission.
      const before = pairs.length;
      svc.setActive('d1', 'm1');
      expect(pairs.length).toBe(before);
    });
  });

  describe('pairKey$ / intentPairKey$', () => {
    it('joins the active halves as `<datasetId>::<modelId>`', () => {
      const keys: string[] = [];
      svc.pairKey$.subscribe((k) => keys.push(k));
      expect(keys[keys.length - 1]).toBe('::');

      svc.setActive('d1', 'm1');
      expect(keys[keys.length - 1]).toBe('d1::m1');
    });

    it('joins the intent halves independently of active', () => {
      const keys: string[] = [];
      svc.intentPairKey$.subscribe((k) => keys.push(k));

      svc.setIntent('dX', 'mY');
      expect(keys[keys.length - 1]).toBe('dX::mY');
    });
  });

  describe('single-half changes', () => {
    it('setActive updates only the changed half', () => {
      svc.setActive('d1', 'm1');
      svc.setActive('d1', 'm2');
      expect(svc.datasetId).toBe('d1');
      expect(svc.modelId).toBe('m2');
    });

    it('setIntent with no change is a no-op (no intentPair emission)', () => {
      svc.setIntent('d1', 'm1');
      const pairs: unknown[] = [];
      svc.intentPair$.subscribe((p) => pairs.push(p));
      expect(pairs.length).toBe(1);
      svc.setIntent('d1', 'm1');
      expect(pairs.length).toBe(1);
    });
  });

  describe('nextRequestId', () => {
    it('increments monotonically and currentRequestId reflects the latest', () => {
      expect(svc.nextRequestId()).toBe(1);
      expect(svc.nextRequestId()).toBe(2);
      expect(svc.currentRequestId).toBe(2);
    });
  });

  describe('mediaUrl', () => {
    it('returns the bare path when no active ids are set', () => {
      expect(svc.mediaUrl('/api/media/foo.wav')).toBe('/api/media/foo.wav');
    });

    it('appends only the dataset param when no detector is active', () => {
      svc.setActive('d1', '');
      expect(svc.mediaUrl('/api/media/foo.wav')).toBe('/api/media/foo.wav?dataset_id=d1');
    });

    it('appends both params when both halves are active', () => {
      svc.setActive('d1', 'm1');
      expect(svc.mediaUrl('/x')).toBe('/x?dataset_id=d1&detector_id=m1');
    });

    it('uses the active pair, not intent (intent may still be in-flight)', () => {
      svc.setIntent('dIntent', 'mIntent');
      svc.setActive('dActive', 'mActive');
      expect(svc.mediaUrl('/x')).toBe('/x?dataset_id=dActive&detector_id=mActive');
    });

    it('url-encodes id values', () => {
      svc.setActive('d 1&x', 'm/1');
      expect(svc.mediaUrl('/x')).toBe('/x?dataset_id=d%201%26x&detector_id=m%2F1');
    });
  });
});
