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
  const root = process.env.PLAYWRIGHT_BROWSERS_PATH;
  if (!root || !existsSync(root)) return null;
  // Prefer the headless shell over full chromium, as Playwright's own headless
  // launch does. The full build takes fontconfig's hinting whatever
  // `--font-render-hinting=none` says, so on a machine whose fontconfig hints
  // (the GRID: `hintstyle` slight) it drifted on 142 of 162 shots where the
  // shell of the same build matched the baselines (#5391). The shell cannot do
  // a headed run, which the harness never asks for. The revision still comes
  // first: an older shell does not beat a newer build.
  const revision = (n) => Number(n.match(/-(\d+)$/)?.[1] ?? 0);
  const shell = (n) => (n.includes('headless_shell') ? 0 : 1);
  const candidates = readdirSync(root)
    .filter((d) => d.startsWith('chromium'))
    .sort((a, b) => revision(b) - revision(a) || shell(a) - shell(b))
    // Older Playwright builds unpack to `chrome-linux/`; current ones to
    // `chrome-linux64/` (and the shell to its own `chrome-headless-shell-linux64/`).
    .flatMap((d) => [
      join(root, d, 'chrome-linux', 'headless_shell'),
      join(root, d, 'chrome-headless-shell-linux64', 'chrome-headless-shell'),
      join(root, d, 'chrome-linux', 'chrome'),
      join(root, d, 'chrome-linux64', 'chrome'),
    ]);
  return candidates.find((p) => existsSync(p)) ?? null;
}

/**
 * Launch chromium: `CHROMIUM_PATH` when set, else the pinned build, else a
 * discovered one. The override wins even where the pinned build is installed,
 * so a machine that has it can still draw in the container's build (#5391).
 */
export async function launchChromium(options = {}) {
  if (process.env.CHROMIUM_PATH) {
    console.log(`[launch] CHROMIUM_PATH: using ${process.env.CHROMIUM_PATH}`);
    return await chromium.launch({ ...options, executablePath: process.env.CHROMIUM_PATH });
  }
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
