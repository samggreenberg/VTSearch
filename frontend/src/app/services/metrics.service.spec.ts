import { TestBed } from '@angular/core/testing';

import { METRICS_PROVIDER, MetricsEvent, MetricsService } from './metrics.service';
import { ServerMetricsProvider } from './server-metrics-provider';
import { provideHttpTesting } from '../testing/test-providers';

describe('MetricsService', () => {
  let sent: MetricsEvent[];
  let service: MetricsService;

  beforeEach(() => {
    sent = [];
    TestBed.configureTestingModule({
      providers: [{ provide: METRICS_PROVIDER, useValue: { send: (e: MetricsEvent) => sent.push(e) } }],
    });
    service = TestBed.inject(MetricsService);
  });

  afterEach(() => {
    document.body.innerHTML = '';
  });

  describe('logEvent', () => {
    it('hands the event to the configured provider', () => {
      service.logEvent({ group: 'resultsExported', attributes: { exporter: 'csv' } });
      expect(sent).toEqual([{ group: 'resultsExported', attributes: { exporter: 'csv' } }]);
    });

    it('defaults attributes to an empty object', () => {
      service.logEvent({ group: 'view' });
      expect(sent).toEqual([{ group: 'view', attributes: {} }]);
    });

    it('copies the attributes, so a later mutation does not reach the provider', () => {
      const attributes = { key: 'theme' };
      service.logEvent({ group: 'settingChanged', attributes });
      attributes.key = 'volume';
      expect(sent[0].attributes).toEqual({ key: 'theme' });
    });
  });

  describe('data-metric clicks', () => {
    function mount(html: string): void {
      document.body.innerHTML = html;
    }

    it('logs a click on a tagged control', () => {
      mount('<button data-metric="imageZoomIn">+</button>');
      document.querySelector('button')!.click();
      expect(sent).toEqual([{ group: 'click', attributes: { target: 'imageZoomIn' } }]);
    });

    it('logs the nearest tagged ancestor of the clicked element', () => {
      mount('<button data-metric="openBrowse"><span class="icon">B</span></button>');
      (document.querySelector('.icon') as HTMLElement).click();
      expect(sent).toEqual([{ group: 'click', attributes: { target: 'openBrowse' } }]);
    });

    it('ignores clicks on untagged elements', () => {
      mount('<button>plain</button>');
      document.querySelector('button')!.click();
      expect(sent).toEqual([]);
    });

    it('still sees a click whose handler stops propagation', () => {
      mount('<div data-metric="panel"><button>x</button></div>');
      const button = document.querySelector('button')!;
      button.addEventListener('click', (e) => e.stopPropagation());
      button.click();
      expect(sent).toEqual([{ group: 'click', attributes: { target: 'panel' } }]);
    });

    it('stops listening once the injector is destroyed', () => {
      mount('<button data-metric="late">x</button>');
      TestBed.resetTestingModule();
      document.querySelector('button')!.click();
      expect(sent).toEqual([]);
    });
  });
});

describe('METRICS_PROVIDER default', () => {
  it('is the server provider when nothing overrides it', () => {
    TestBed.configureTestingModule({ providers: [...provideHttpTesting()] });
    expect(TestBed.inject(METRICS_PROVIDER)).toBe(TestBed.inject(ServerMetricsProvider));
  });
});
