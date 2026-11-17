import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { join, relative } from 'node:path';

/**
 * Every place a template draws a bound image either blurs it or says why not
 * (#5696).
 *
 * There is no shared thumbnail component: about twenty templates each write
 * their own `<img>`, so a site added later could draw a dataset image sharp
 * whatever its blur state says. This reads every template and fails on an
 * `<img>` whose source is bound (or an element whose background image is)
 * that carries no `vtBlur`, unless {@link ALLOWED} names it with a reason.
 *
 * A template alone cannot say whether a bound URL is a dataset image (the URL
 * is built in the component, or in a parent, as the vote grid's is), so the
 * rule is every bound image, and the allowlist holds the ones that are not
 * dataset items, or that blur another way.
 */

interface Allowed {
  /** The template, relative to `src/app`. */
  readonly file: string;
  /** Text in the element's own tag that singles it out. */
  readonly match: string;
  readonly why: string;
}

const ALLOWED: readonly Allowed[] = [
  {
    file: 'components/toast-container/toast-container.component.html',
    match: 'toastyFace(',
    why: "Toasty's face, a bundled asset",
  },
  {
    file: 'components/toasty-hint/toasty-hint.component.html',
    match: 'faceSrc()',
    why: "Toasty's face, a bundled asset",
  },
  {
    file: 'components/dashboard/new-detector-modal/new-detector-modal.component.html',
    match: 'class="example-thumbnail"',
    why: 'an uploaded example file (/api/server-media-files/…), not a dataset item',
  },
  {
    file: 'components/center-panel/image-viewer/image-viewer.component.html',
    match: 'imageSrc()',
    why: "the focused viewer composes the blur into its own `imageFilter` (#5695): `.vt-blurred`'s filter would drop its tone map",
  },
];

/** One element's opening tag, quoted attribute values kept whole (an expression may hold `>`). */
const TAG = /<([a-zA-Z][\w-]*)\b(?:[^>"']|"[^"]*"|'[^']*')*>/g;
const BOUND_IMG_SRC = /\[(?:attr\.)?(?:src|srcset)\]\s*=/;
const BOUND_BACKGROUND = /\[style\.(?:background-image|backgroundImage)\]\s*=/;
const HAS_DIRECTIVE = /\[vtBlur\]\s*=|\svtBlur(?:\s|=|>|\/)/;

const APP_DIR = join(process.cwd(), 'src', 'app');

function templates(dir: string): string[] {
  const out: string[] = [];
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const path = join(dir, entry.name);
    if (entry.isDirectory()) {
      if (entry.name !== 'generated') out.push(...templates(path));
    } else if (entry.name.endsWith('.html')) {
      out.push(path);
    }
  }
  return out;
}

/** Each bound image without `vtBlur`, as `{file, tag}`. */
function unblurredImages(): { file: string; tag: string }[] {
  const found: { file: string; tag: string }[] = [];
  for (const path of templates(APP_DIR)) {
    const file = relative(APP_DIR, path).split('\\').join('/');
    for (const m of readFileSync(path, 'utf8').matchAll(TAG)) {
      const [tag, name] = m;
      const drawsImage = (name.toLowerCase() === 'img' && BOUND_IMG_SRC.test(tag)) || BOUND_BACKGROUND.test(tag);
      if (drawsImage && !HAS_DIRECTIVE.test(tag)) found.push({ file, tag: tag.replace(/\s+/g, ' ') });
    }
  }
  return found;
}

function allowedBy(image: { file: string; tag: string }): Allowed | undefined {
  return ALLOWED.find((a) => a.file === image.file && image.tag.includes(a.match));
}

describe('vtBlur coverage of image sites (#5696)', () => {
  it('runs from the frontend directory, where it can read the templates', () => {
    expect(existsSync(APP_DIR)).toBe(true);
    expect(templates(APP_DIR).length).toBeGreaterThan(50);
  });

  it('puts vtBlur on every bound image a template draws, or allows it with a reason', () => {
    const missing = unblurredImages()
      .filter((image) => !allowedBy(image))
      .map(({ file, tag }) => `${file}: ${tag}`);
    // A new image site: add `[vtBlur]="<media id>"` (plus `[vtBlurDataset]`
    // for another dataset's item), or, if it draws no dataset item, an
    // ALLOWED entry saying so.
    expect(missing).toEqual([]);
  });

  it('allows nothing that is not there any more', () => {
    const images = unblurredImages();
    const stale = ALLOWED.filter((a) => !images.some((image) => allowedBy(image) === a)).map((a) => `${a.file}: ${a.match}`);
    expect(stale).toEqual([]);
  });

  it('catches an image bound without the directive, and passes one with it', () => {
    const scan = (html: string): boolean[] =>
      Array.from(html.matchAll(TAG)).map(
        ([tag, name]) =>
          ((name === 'img' && BOUND_IMG_SRC.test(tag)) || BOUND_BACKGROUND.test(tag)) && !HAS_DIRECTIVE.test(tag),
      );
    expect(scan('<img class="a" [src]="url(x > 1 ? \'y\' : \'z\')" alt="" />')).toEqual([true]);
    expect(scan('<img\n  [src]="url()"\n  [vtBlur]="id"\n  alt="" />')).toEqual([false]);
    expect(scan('<img src="logo.png" alt="" />')).toEqual([false]);
    expect(scan('<div [style.background-image]="bg()"></div>')).toEqual([true]);
  });
});
