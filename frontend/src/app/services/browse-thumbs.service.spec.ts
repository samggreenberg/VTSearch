import { TestBed } from '@angular/core/testing';
import { BrowseThumbsService, RETAINED_THUMBS, type ThumbHost } from './browse-thumbs.service';
import { ActiveContextService } from './active-context.service';
import { makeActiveContextStub } from '../testing/mocks';

describe('BrowseThumbsService (#5637)', () => {
  let service: BrowseThumbsService;

  const host = (): ThumbHost => ({ onLoaded: vi.fn(), accent: () => '#00f' });

  beforeEach(() => {
    vi.unstubAllGlobals();
    // A thumbnail fetch here would need a browser; stub Image so `get` only
    // records what it asked for.
    vi.stubGlobal(
      'Image',
      class {
        src = '';
        decoding = '';
        complete = false;
        naturalWidth = 0;
        onload: (() => void) | null = null;
        onerror: (() => void) | null = null;
        setAttribute() {}
        addEventListener() {}
      },
    );
    TestBed.configureTestingModule({
      providers: [{ provide: ActiveContextService, useValue: makeActiveContextStub() }],
    });
    service = TestBed.inject(BrowseThumbsService);
  });

  afterEach(() => vi.unstubAllGlobals());

  it('hands every canvas the same store, so a revisit finds the last visit\'s thumbnails', () => {
    const first = host();
    const store = service.attach(first);
    service.useDataset('ds1');
    store.get(7);
    service.detach(first);

    const second = host();
    expect(service.attach(second)).toBe(store);
    service.useDataset('ds1');
    expect(store.peek(7)).toBeDefined();
  });

  it('drops the cache for another dataset, whose media ids mean other items', () => {
    const store = service.attach(host());
    service.useDataset('ds1');
    store.get(7);
    service.useDataset('ds2');
    expect(store.size).toBe(0);
  });

  it('repaints through whichever canvas is attached', () => {
    const first = host();
    service.attach(first);
    service.detach(first);
    const second = host();
    service.attach(second);
    // The store's repaint goes to the canvas attached now.
    (service.store as unknown as { deps: { onLoaded(): void } }).deps.onLoaded();
    expect(second.onLoaded).toHaveBeenCalledTimes(1);
    expect(first.onLoaded).not.toHaveBeenCalled();
  });

  it('trims to the retained count when its canvas goes away', () => {
    const first = host();
    const store = service.attach(first);
    service.useDataset('ds1');
    for (let id = 0; id < RETAINED_THUMBS + 10; id++) store.get(id);
    service.detach(first);
    expect(store.size).toBe(RETAINED_THUMBS);
  });

  it('leaves the store alone when an old canvas leaves after a new one arrived', () => {
    const first = host();
    const store = service.attach(first);
    service.useDataset('ds1');
    for (let id = 0; id < RETAINED_THUMBS + 10; id++) store.get(id);
    const second = host();
    service.attach(second);
    service.detach(first);
    expect(store.size).toBe(RETAINED_THUMBS + 10);
  });
});
