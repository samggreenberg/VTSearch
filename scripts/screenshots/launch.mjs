/**
 * Shared chromium launcher for the screenshot harness.
 *
 * Playwright looks for a browser build whose revision is pinned by the
 * `playwright` version in this folder's `package.json`. That pin is a caret
 * range, so a fresh `npm install` floats to whatever is current and then
 * demands a browser revision that may not be on the machine — while a
 * perfectly usable chromium sits right there under `PLAYWRIGHT_BROWSERS_PATH`.
 * That is exactly the state of the Claude-Code-on-the-web container, which
 * ships a chromium but not necessarily *this* week's revision, and it made the
 * harness look like "no browser here" when the truth was "wrong revision".
 *
 * So: try the pinned browser first (the right answer on a dev box that ran
 * `npx playwright install`), and fall back to any chromium found in the
 * browsers directory. `CHROMIUM_PATH` overrides both.
 */
import { chromium } from 'playwright';
import { existsSync, readdirSync } from 'node:fs';
import { join } from 'node:path';

/** Any chromium executable under PLAYWRIGHT_BROWSERS_PATH, newest revision first. */
function discoverChromium() {
  if (process.env.CHROMIUM_PATH) return process.env.CHROMIUM_PATH;
  const root = process.env.PLAYWRIGHT_BROWSERS_PATH;
  if (!root || !existsSync(root)) return null;
  // Prefer full chromium over the headless shell: the shell cannot do a headed
  // run, and some UI surfaces (popups, downloads) behave closer to a user's
  // browser in the full build.
  const candidates = readdirSync(root)
    .filter((d) => d.startsWith('chromium'))
    .sort((a, b) => {
      const shell = (n) => (n.includes('headless_shell') ? 1 : 0);
      if (shell(a) !== shell(b)) return shell(a) - shell(b);
      return b.localeCompare(a, undefined, { numeric: true });
    })
    .flatMap((d) => [
      join(root, d, 'chrome-linux', 'chrome'),
      join(root, d, 'chrome-linux', 'headless_shell'),
    ]);
  return candidates.find((p) => existsSync(p)) ?? null;
}

/** Launch chromium, falling back to a discovered build if the pinned one is absent. */
export async function launchChromium(options = {}) {
  try {
    return await chromium.launch(options);
  } catch (err) {
    if (!/Executable doesn't exist/i.test(String(err?.message))) throw err;
    const executablePath = discoverChromium();
    if (!executablePath) throw err;
    console.log(`[launch] pinned browser missing; using ${executablePath}`);
    return await chromium.launch({ ...options, executablePath });
  }
}
