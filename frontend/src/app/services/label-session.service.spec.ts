import { TestBed } from '@angular/core/testing';
import { LabelSessionService } from './label-session.service';

describe('LabelSessionService', () => {
  let service: LabelSessionService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(LabelSessionService);
  });

  describe('mediaExampleFilenames', () => {
    it('returns every media example from the examples list', () => {
      service.examples = [
        { type: 'media', value: 'a.jpg' },
        { type: 'text', value: 'a red car' },
        { type: 'media', value: 'b.jpg' },
      ];
      service.mediaExample = 'a.jpg';

      expect(service.mediaExampleFilenames).toEqual(['a.jpg', 'b.jpg']);
    });

    it('falls back to the scalar mediaExample when the list has no media entries', () => {
      service.examples = [{ type: 'text', value: 'a red car' }];
      service.mediaExample = 'legacy.jpg';

      expect(service.mediaExampleFilenames).toEqual(['legacy.jpg']);
    });

    it('returns empty when neither the list nor the scalar is set', () => {
      service.examples = [];
      service.mediaExample = '';

      expect(service.mediaExampleFilenames).toEqual([]);
    });

    it('skips media entries with empty values', () => {
      service.examples = [{ type: 'media', value: '' }];
      service.mediaExample = '';

      expect(service.mediaExampleFilenames).toEqual([]);
    });
  });
});
