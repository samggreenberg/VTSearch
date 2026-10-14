/**
 * Toasty telling the user to stop, drawn over a frame of the train loop.
 *
 *   node slides/figs/src/stop-hint.mjs    # figs/ui-train-loop.build6.webp -> figs/ui-train-loop.webp
 *
 * Rock the Vote's last page is the app saying the detector is trained, so the
 * room hears that VTSearch tells you when to stop and you never have to decide
 * it yourself (#4833). The real hand-off (#4680) only fires once every quality
 * light is green, which a session reaches at a median of eighty-odd votes, and
 * the deck's session is thirty-one votes long: it is the session Find runs, and
 * voting on would change every frame after it. So this page is staged. The
 * owner asked for it that way (#4833: "I don't need it to be ML perfect"); it
 * is the one frame in the train loop that is not the app's own.
 *
 * Staged, but drawn the way the app draws it. The bubble is `vt-toasty-hint`'s
 * markup and styles (`frontend/src/app/components/toasty-hint/`), carrying the
 * `autopilot-done` hint's words (`frontend/src/app/app.component.html`), and it
 * is placed by the component's own rule, `toastyHintPlacement`: Toasty centred
 * under the Dashboard button, `GAP` below it, the bubble reaching right. Every
 * colour and size reads the app's theme token, with the light theme's value as
 * the fallback, so on a live page the app's own tokens win.
 *
 * Two callers draw it with the same function:
 *
 *   - `shoot-ui-figs.mjs` on the live app, right after it shoots the session's
 *     last real frame as `ui-train-loop.build6`, so a reshoot redraws it;
 *   - this file's CLI, on that committed frame, with an invisible stand-in for
 *     the Dashboard button laid where the frame shows it. That is how the page
 *     was first made without re-running a session, and how to redraw it after
 *     changing the words or the look without re-running one either.
 *
 * Keep the words and the styles in step with the component if either changes.
 */
import { launchChromium } from '../../../scripts/screenshots/launch.mjs';
import { resolveBox } from '../../../scripts/screenshots/callouts.mjs';
import { execFileSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO = resolve(HERE, '../../..');
const FIGS = resolve(HERE, '..');

const LAYER_ID = '__stop_hint';

/** The control Toasty stands at: the top bar's Dashboard button. */
export const STOP_ANCHOR = { selector: 'button.top-bar-btn', hasText: 'Dashboard' };

/** The `autopilot-done` hint, as `app.component.html` words it. */
const WORDS =
  'Your detector is trained! Every quality light is green. Keep voting to sharpen it, or '
  + 'click <strong>Dashboard</strong> to test it or put it to work.';

/** `toastyHintPlacement`'s GAP and Toasty's `--toasty-size`, in CSS px. */
const GAP = 4;
const TOASTY = 64;

/** `toasty-hint.component.scss`, scoped to the layer, tokens with light-theme fallbacks. */
const STYLE = `
#${LAYER_ID} {
  position: fixed; z-index: 2147483000; width: max-content;
  display: flex; flex-direction: row; align-items: flex-start;
  gap: var(--space-xs, 4px); pointer-events: none;
}
#${LAYER_ID} img { flex-shrink: 0; width: ${TOASTY}px; height: ${TOASTY}px; }
#${LAYER_ID} .bubble {
  position: relative; display: flex; flex-direction: column; gap: var(--space-sm, 6px);
  max-width: 260px; margin-top: var(--space-sm, 6px); padding: var(--space-md, 8px);
  background: var(--bg-surface, #ffffff); border: 1px solid var(--accent, #5a68d8);
  border-radius: var(--radius-lg, 6px);
  box-shadow: 0 4px 12px var(--shadow-dropdown, rgba(0, 0, 0, 0.15));
}
#${LAYER_ID} .bubble::after {
  content: ''; position: absolute; top: var(--space-lg, 12px); left: -6px;
  width: 10px; height: 10px; background: inherit;
  border: solid var(--accent, #5a68d8); border-width: 0 0 1px 1px; transform: rotate(45deg);
}
#${LAYER_ID} p {
  margin: 0; font-size: var(--font-md, 14px); color: var(--text-primary, #1a1a2e);
}
#${LAYER_ID} .options { display: flex; flex-wrap: wrap; gap: var(--space-xs, 4px) var(--space-md, 8px); }
#${LAYER_ID} label {
  display: inline-flex; align-items: center; gap: var(--space-xs, 4px);
  font-size: var(--font-sm, 13px); color: var(--text-muted, #666);
}
`;

/**
 * Draw Toasty's "trained" hint under the Dashboard button on *page*.
 *
 * `face` is the URL of Toasty's picture as the page can load it: the app
 * serves him at `logo.png`, which is the default.
 */
export async function drawStopHint(page, { face = 'logo.png' } = {}) {
  const anchor = await resolveBox(page, STOP_ANCHOR);
  if (!anchor) throw new Error('stop hint: no Dashboard button to stand Toasty under');
  await page.evaluate(
    ({ id, style, words, face, left, top }) => {
      document.getElementById(id)?.remove();
      const host = document.createElement('div');
      host.id = id;
      host.style.left = `${left}px`;
      host.style.top = `${top}px`;
      host.innerHTML =
        `<style>${style}</style><img alt="" src="${face}">`
        + `<div class="bubble"><p>${words}</p><div class="options">`
        + '<label><input type="checkbox"> Hide this hint</label>'
        + '<label><input type="checkbox"> Hide all hints</label>'
        + '</div></div>';
      document.body.appendChild(host);
      const img = host.querySelector('img');
      return img.complete ? null : new Promise((ok, fail) => {
        img.onload = ok;
        img.onerror = () => fail(new Error(`stop hint: Toasty's picture did not load from ${face.slice(0, 80)}`));
      });
    },
    {
      id: LAYER_ID,
      style: STYLE,
      words: WORDS,
      face,
      left: anchor.x + anchor.w / 2 - TOASTY / 2,
      top: anchor.y + anchor.h + GAP,
    }
  );
}

/** Take the hint back off. */
export async function clearStopHint(page) {
  await page.evaluate((id) => document.getElementById(id)?.remove(), LAYER_ID);
}

// ── the CLI: redraw the committed frame ──────────────────────────────────────

/**
 * Where the app sits in a composed frame, and the Dashboard button in the app,
 * in the app's CSS px. `shoot-ui-figs.mjs` shoots a 1180x830 window at 2x and
 * pads it to 16:9 (`compose`), which puts the window 508 CSS px from the
 * frame's left and 73 from its top; the button's box is read off the frame.
 * Re-measure both if the harness's window, margins or top bar move.
 */
const FRAME = { width: 1735, height: 976, scale: 2 };
const WINDOW = { left: 508, top: 73 };
const BUTTON = { left: 100, top: 10, width: 86, height: 25 };

/** *path* as a data URL: a page set from a string may not load `file://`. */
const dataUrl = (path, type) => `data:${type};base64,${readFileSync(path).toString('base64')}`;

async function redrawCommitted() {
  const src = dataUrl(join(FIGS, 'ui-train-loop.build6.webp'), 'image/webp');
  const out = join(FIGS, 'ui-train-loop.webp');
  const browser = await launchChromium();
  try {
    const page = await browser.newPage({
      viewport: { width: FRAME.width, height: FRAME.height },
      deviceScaleFactor: FRAME.scale,
    });
    // The frames were shot on a machine whose sans-serif is Arial's metrics,
    // so the bubble is set in Liberation Sans to sit with the app around it.
    await page.setContent(`<!doctype html><html><body style="margin:0;
        font-family: 'Liberation Sans', Arial, sans-serif;
        width:${FRAME.width}px; height:${FRAME.height}px;
        background: url('${src}') 0 0 / 100% 100% no-repeat">
      <button class="top-bar-btn" style="position:absolute; opacity:0; padding:0; border:0;
        left:${WINDOW.left + BUTTON.left}px; top:${WINDOW.top + BUTTON.top}px;
        width:${BUTTON.width}px; height:${BUTTON.height}px">Dashboard</button>
      </body></html>`);
    await page.waitForFunction(() => document.readyState === 'complete');
    await drawStopHint(page, {
      face: dataUrl(join(REPO, 'frontend', 'public', 'logo.png'), 'image/png'),
    });
    const png = await page.screenshot({ type: 'png' });
    execFileSync(
      'python',
      [
        '-c',
        'import sys;from io import BytesIO;from PIL import Image;'
          + 'Image.open(BytesIO(sys.stdin.buffer.read())).convert("RGB")'
          + '.save(sys.argv[1],"WEBP",quality=92,method=6)',
        out,
      ],
      { cwd: REPO, input: png, stdio: ['pipe', 'inherit', 'inherit'] }
    );
    console.log(`[stop-hint] wrote figs/ui-train-loop.webp`);
  } finally {
    await browser.close();
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  await redrawCommitted();
}
