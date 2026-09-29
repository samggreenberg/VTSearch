import { signal, WritableSignal } from '@angular/core';
import { ComponentFixture, TestBed } from '@angular/core/testing';

import { HttpTestingController } from '@angular/common/http/testing';
import { KeyboardHelpModalComponent, headingSlug, resolveDocPath } from './keyboard-help-modal.component';
import { configureZoneless } from '../../../testing/zoneless-testbed';
import { provideHttpTesting } from '../../../testing/test-providers';
import { SettingsStateService } from '../../../services/settings-state.service';
import type { AppSettings } from '../../../generated/api-client/models/app-settings';

/**
 * The guide's table of contents is written against GitHub's heading anchors,
 * but `marked` v14 emits bare `<h2>`s — so the in-app copy has to slug the
 * headings itself or every TOC entry lands nowhere. These specs pin the slug
 * rule and the injection, which is the whole of that contract.
 */
describe('KeyboardHelpModalComponent — in-app guide anchors', () => {
  let component: KeyboardHelpModalComponent;
  let fixture: ComponentFixture<KeyboardHelpModalComponent>;
  let httpMock: HttpTestingController;

  const GUIDE_URL = 'assets/docs/USER_GUIDE.md';

  beforeEach(async () => {
    await configureZoneless({
      imports: [KeyboardHelpModalComponent],
      providers: [...provideHttpTesting()],
    }).compileComponents();

    fixture = TestBed.createComponent(KeyboardHelpModalComponent);
    component = fixture.componentInstance;
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => {
    httpMock.verify();
  });

  /** Open the guide tab and answer its fetch with *markdown*. */
  async function loadGuide(markdown: string): Promise<HTMLElement> {
    component.selectTab('guide');
    httpMock.expectOne(GUIDE_URL).flush(markdown);
    await fixture.whenStable();
    return fixture.nativeElement.querySelector('.guide-body') as HTMLElement;
  }

  it('gives every heading a GitHub-compatible id', async () => {
    const body = await loadGuide(
      ['# VTSearch User Guide', '', '## Autopilot: the guided workflow', '', '### Pre-computed embeddings (.npz)'].join(
        '\n',
      ),
    );

    expect(body.querySelector('h1')?.id).toBe('vtsearch-user-guide');
    expect(body.querySelector('h2')?.id).toBe('autopilot-the-guided-workflow');
    expect(body.querySelector('h3')?.id).toBe('pre-computed-embeddings-npz');
  });

  it('resolves the table of contents against the headings it renders', async () => {
    const body = await loadGuide(
      [
        '1. [Autopilot: the guided workflow](#autopilot-the-guided-workflow)',
        '',
        '## Autopilot: the guided workflow',
      ].join('\n'),
    );

    const href = body.querySelector('a')?.getAttribute('href');
    expect(href).toBe('#autopilot-the-guided-workflow');
    expect(body.querySelector(`h2[id="${href!.slice(1)}"]`)).toBeTruthy();
  });

  it('disambiguates repeated headings the way GitHub does', async () => {
    const body = await loadGuide(['## Stats', '', '## Stats'].join('\n'));

    const ids = Array.from(body.querySelectorAll('h2')).map((h) => h.id);
    expect(ids).toEqual(['stats', 'stats-1']);
  });

  it('handles an anchor click itself instead of navigating', async () => {
    const body = await loadGuide(['[Go](#target-section)', '', '## Target section'].join('\n'));

    const link = body.querySelector('a') as HTMLAnchorElement;
    const event = new MouseEvent('click', { bubbles: true, cancelable: true });
    link.dispatchEvent(event);

    expect(event.defaultPrevented).toBe(true);
  });

  it('keeps an inline crop in its sentence, sized by height alone', async () => {
    // The guide sets small crops of buttons into running text (#4202). The
    // stylesheet tells them from full screenshots by "height, no width", so
    // both the placement and the attribute have to survive rendering and
    // sanitisation.
    const body = await loadGuide(
      'Click <picture><source media="(prefers-color-scheme: dark)" srcset="assets/icon-train.dark.webp" />' +
        '<img src="assets/icon-train.light.webp" alt="The Train button" height="24" /></picture> to start.',
    );

    const img = body.querySelector('img') as HTMLImageElement;
    expect(img.closest('p')?.textContent).toContain('to start.');
    expect(img.getAttribute('height')).toBe('24');
    expect(img.hasAttribute('width')).toBe(false);
    expect(img.getAttribute('src')).toMatch(/^assets\/docs\/assets\/icon-train\.(light|dark)\.webp$/);
  });

  it('leaves non-anchor links alone', async () => {
    const body = await loadGuide('[Docs](https://example.com/docs)');

    // Read `defaultPrevented` from a document-level listener — it bubbles past
    // `.guide-body`, so it sees the component's verdict — then cancel the event
    // so jsdom doesn't try to actually navigate to example.com.
    let prevented: boolean | null = null;
    const observe = (event: Event): void => {
      prevented = event.defaultPrevented;
      event.preventDefault();
    };
    document.addEventListener('click', observe);
    try {
      const link = body.querySelector('a') as HTMLAnchorElement;
      link.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
    } finally {
      document.removeEventListener('click', observe);
    }

    expect(prevented).toBe(false);
  });
});

/**
 * The how-to pages under docs/user/howto/ are linked from the guide. On GitHub
 * those links just work; in the Help panel they must open the page in the same
 * pane (a plain navigation would leave the SPA for a raw .md file), with its
 * `../assets/` screenshots resolved against the page, and Back to return.
 */
describe('KeyboardHelpModalComponent — how-to pages in the guide pane', () => {
  let component: KeyboardHelpModalComponent;
  let fixture: ComponentFixture<KeyboardHelpModalComponent>;
  let httpMock: HttpTestingController;

  beforeEach(async () => {
    await configureZoneless({
      imports: [KeyboardHelpModalComponent],
      providers: [...provideHttpTesting()],
    }).compileComponents();

    fixture = TestBed.createComponent(KeyboardHelpModalComponent);
    component = fixture.componentInstance;
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => {
    httpMock.verify();
  });

  const body = (): HTMLElement => fixture.nativeElement.querySelector('.guide-body') as HTMLElement;

  /** Click the first link in the guide pane whose href is *href*; report whether the component took it. */
  function clickLink(href: string): boolean {
    // Read the verdict from a document-level listener (it bubbles past
    // `.guide-body`), then cancel so jsdom doesn't attempt a real navigation.
    let taken: boolean | null = null;
    const observe = (event: Event): void => {
      taken = event.defaultPrevented;
      event.preventDefault();
    };
    document.addEventListener('click', observe);
    try {
      const link = body().querySelector(`a[href="${href}"]`) as HTMLAnchorElement;
      link.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
    } finally {
      document.removeEventListener('click', observe);
    }
    return taken === true;
  }

  async function openGuide(markdown: string): Promise<void> {
    component.selectTab('guide');
    httpMock.expectOne('assets/docs/USER_GUIDE.md').flush(markdown);
    await fixture.whenStable();
  }

  it('opens a linked how-to page in the pane, and Back returns to the guide', async () => {
    await openGuide(['# VTSearch User Guide', '', '[Fix the calls](howto/fix-find-results.md)'].join('\n'));
    expect(fixture.nativeElement.querySelector('.back-btn')).toBeNull();

    expect(clickLink('howto/fix-find-results.md')).toBe(true);
    httpMock.expectOne('assets/docs/howto/fix-find-results.md').flush('# Fix the calls\n\n[Guide](../USER_GUIDE.md)');
    await fixture.whenStable();

    expect(body().querySelector('h1')?.textContent).toBe('Fix the calls');
    const back = fixture.nativeElement.querySelector('.back-btn') as HTMLButtonElement;
    expect(back?.textContent).toContain('Back');

    back.click();
    await fixture.whenStable();
    // Cached: Back re-renders without a second fetch (httpMock.verify checks).
    expect(body().querySelector('h1')?.textContent).toBe('VTSearch User Guide');
    expect(fixture.nativeElement.querySelector('.back-btn')).toBeNull();
  });

  it("resolves a how-to page's screenshots against the page, not the guide", async () => {
    await openGuide('[Page](howto/page.md)');
    clickLink('howto/page.md');
    httpMock
      .expectOne('assets/docs/howto/page.md')
      .flush('<img src="../assets/step-find.light.webp" alt="Step" width="720" />');
    await fixture.whenStable();

    expect(body().querySelector('img')?.getAttribute('src')).toMatch(/^assets\/docs\/assets\/step-find\.(light|dark)\.webp$/);
  });

  it('follows a link back into the guide at its heading', async () => {
    await openGuide(['[Page](howto/page.md)', '', '## Autopilot: the guided workflow'].join('\n'));
    clickLink('howto/page.md');
    httpMock.expectOne('assets/docs/howto/page.md').flush('[Autopilot](../USER_GUIDE.md#autopilot-the-guided-workflow)');
    await fixture.whenStable();

    expect(clickLink('../USER_GUIDE.md#autopilot-the-guided-workflow')).toBe(true);
    await fixture.whenStable();
    expect(body().querySelector('h2')?.id).toBe('autopilot-the-guided-workflow');
  });

  it('leaves a link out of docs/user/ to the browser', async () => {
    await openGuide('[Setup](../SETUP.md)');
    expect(clickLink('../SETUP.md')).toBe(false);
  });
});

/**
 * A deployment can list its own docs (plugin guides, a lab wiki) through the
 * server's ``docs_links`` setting (#4310). They sit in the footer, so they
 * show on every tab, and each opens in a new tab.
 */
describe('KeyboardHelpModalComponent — server doc links', () => {
  let fixture: ComponentFixture<KeyboardHelpModalComponent>;
  let settingsSignal: WritableSignal<AppSettings | null>;

  beforeEach(async () => {
    settingsSignal = signal<AppSettings | null>(null);
    await configureZoneless({
      imports: [KeyboardHelpModalComponent],
      providers: [...provideHttpTesting(), { provide: SettingsStateService, useValue: { settingsSignal } }],
    }).compileComponents();

    fixture = TestBed.createComponent(KeyboardHelpModalComponent);
  });

  function docLinks(): HTMLAnchorElement[] {
    return Array.from(fixture.nativeElement.querySelectorAll('.help-docs a'));
  }

  it('shows no docs block when the server lists none', async () => {
    settingsSignal.set({ docs_links: [] });
    await fixture.whenStable();

    expect(fixture.nativeElement.querySelector('.help-docs')).toBeNull();
    // The "Email us" line is unaffected.
    expect(fixture.nativeElement.querySelector('.help-footer__contact a')?.getAttribute('href')).toMatch(/^mailto:/);
  });

  it('shows no docs block before settings load', async () => {
    await fixture.whenStable();
    expect(fixture.nativeElement.querySelector('.help-docs')).toBeNull();
  });

  it("lists the server's links in its order, each opening in a new tab", async () => {
    settingsSignal.set({
      docs_links: [
        { label: 'Acme plugin guide', url: 'https://acme.example/docs' },
        { label: 'Lab wiki', url: '/wiki/vtsearch' },
      ],
    });
    await fixture.whenStable();

    const links = docLinks();
    expect(links.map((a) => a.getAttribute('href'))).toEqual(['https://acme.example/docs', '/wiki/vtsearch']);
    expect(links[0].textContent).toContain('Acme plugin guide');
    expect(links[1].textContent).toContain('Lab wiki');
    for (const a of links) {
      expect(a.getAttribute('target')).toBe('_blank');
      expect(a.getAttribute('rel')).toBe('noopener noreferrer');
    }
  });

  it('keeps the links on the User guide tab too', async () => {
    settingsSignal.set({ docs_links: [{ label: 'Acme plugin guide', url: 'https://acme.example/docs' }] });
    fixture.componentInstance.selectTab('guide');
    TestBed.inject(HttpTestingController).expectOne('assets/docs/USER_GUIDE.md').flush('# Guide');
    await fixture.whenStable();

    expect(docLinks().length).toBe(1);
  });
});

describe('resolveDocPath', () => {
  it('resolves relative to the linking doc', () => {
    expect(resolveDocPath('USER_GUIDE.md', 'howto/a.md')).toBe('howto/a.md');
    expect(resolveDocPath('howto/a.md', 'b.md#step-2')).toBe('howto/b.md');
    expect(resolveDocPath('howto/a.md', '../USER_GUIDE.md')).toBe('USER_GUIDE.md');
    expect(resolveDocPath('howto/a.md', '../assets/x.light.webp')).toBe('assets/x.light.webp');
  });

  it('refuses what the pane cannot serve', () => {
    expect(resolveDocPath('USER_GUIDE.md', '../SETUP.md')).toBeNull();
    expect(resolveDocPath('USER_GUIDE.md', '#anchor')).toBeNull();
    expect(resolveDocPath('USER_GUIDE.md', 'https://example.com/a.md')).toBeNull();
    expect(resolveDocPath('USER_GUIDE.md', '/abs.md')).toBeNull();
    expect(resolveDocPath('USER_GUIDE.md', 'mailto:someone@example.com')).toBeNull();
  });
});

describe('headingSlug', () => {
  it('turns each space into a hyphen, so a spaced hyphen yields three', () => {
    // The exact bug that killed the guide's TOC: ` - ` is not `-`.
    expect(headingSlug('Autopilot - the guided workflow')).toBe('autopilot---the-guided-workflow');
  });

  it('drops punctuation rather than transliterating it', () => {
    expect(headingSlug('Find: scoring and verifying')).toBe('find-scoring-and-verifying');
    expect(headingSlug('Archive members, no extraction (WebDataset shards)')).toBe(
      'archive-members-no-extraction-webdataset-shards',
    );
  });
});
