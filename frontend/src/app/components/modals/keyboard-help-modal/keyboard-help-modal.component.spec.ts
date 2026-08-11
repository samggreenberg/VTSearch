import { ComponentFixture, TestBed } from '@angular/core/testing';

import { HttpTestingController } from '@angular/common/http/testing';
import { KeyboardHelpModalComponent, headingSlug } from './keyboard-help-modal.component';
import { configureZoneless } from '../../../testing/zoneless-testbed';
import { provideHttpTesting } from '../../../testing/test-providers';

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
