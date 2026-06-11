/**
 * Idempotently create the deterministic fixtures the screenshot recipes need,
 * by driving the running app's UI (so the fixtures look exactly as a user's
 * would). Safe to re-run: each step is skipped if its artifact already exists.
 * refresh.sh runs this before capture.ts. See docs/plans/user-docs-screenshots.md.
 *
 *   - syn-imgs   : 60 synthetic images, SigLIP embedder (the main fixture)
 *   - doc-demo   : an image detector trained on 5 good / 4 bad votes over syn-imgs
 *                  (gives a trained detector for dashboard / results-grid /
 *                   autopilot-phase shots). NB: doc-demo is a throwaway; the
 *                   harness never touches a user's real detectors.
 *   - syn-patch  : 24 synthetic images, DINOv2-patch embedder (region-voting,
 *                  which needs a patch-region-aware embedder)
 *
 * Usage:  node ensure-fixtures.mjs   (APP env overrides the URL)
 */
import { chromium } from 'playwright';

const APP = process.env.APP || 'http://localhost:5000';
const log = (...a) => console.log('[fixtures]', ...a);

const browser = await chromium.launch();
try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  const goDash = async () => {
    await page.goto(`${APP}/#/dashboard`, { waitUntil: 'domcontentloaded' });
    await page.waitForSelector('.dash-table', { timeout: 30000 });
    await page.waitForTimeout(1200);
  };
  const hasDataset = (name) => page.locator('vt-dataset-card', { hasText: name }).count();
  const hasDetector = (name) => page.locator('vt-detector-card', { hasText: name }).count();

  const importSynthetic = async (name, size, embedderLabel) => {
    await page.locator('button[title="Import a new dataset"]').click();
    await page.waitForSelector('.importer-tab-bar', { timeout: 15000 });
    await page.locator('.importer-tab', { hasText: 'Demo' }).click();
    await page.waitForTimeout(600);
    await page.locator('.importer-subtab', { hasText: 'Synthetic Media' }).click();
    await page.waitForSelector('#field-size', { timeout: 10000 });
    await page.fill('#field-size', String(size));
    await page.fill('#field-dataset_name', name);
    if (embedderLabel) {
      await page.locator('button', { hasText: /Advanced/i }).first().click();
      await page.waitForTimeout(400);
      await page.selectOption('#import-advanced-embedder', { label: embedderLabel });
      await page.waitForTimeout(300);
    }
    await page.getByRole('button', { name: 'Import', exact: true }).click();
    // wait for the row to finish embedding (progress text gone)
    await page.waitForFunction((n) => {
      const c = [...document.querySelectorAll('vt-dataset-card')].find((e) => e.textContent.includes(n));
      return c && !/Embedding|Loading dataset/.test(c.textContent);
    }, name, { timeout: 300000 });
    log(`imported ${name}`);
  };

  // 1. syn-imgs (SigLIP)
  await goDash();
  if (await hasDataset('syn-imgs')) log('syn-imgs exists, skipping');
  else await importSynthetic('syn-imgs', 60, null);

  // 2. doc-demo detector (blank, image) + a few votes so it is trained
  await goDash();
  if (await hasDetector('doc-demo')) {
    log('doc-demo exists, skipping');
  } else {
    await page.locator('button[title="Create a new detector"]').click();
    await page.waitForTimeout(800);
    await page.locator('input[placeholder*="dog barking" i]').fill('colorful geometric pattern');
    await page.waitForTimeout(300);
    await page.locator('#detector-name').fill('doc-demo');
    await page.getByRole('button', { name: 'Create', exact: true }).click();
    await page.waitForTimeout(2000);
    log('created doc-demo');
    // Train it: enter the label view and vote 5 good / 4 bad.
    await goDash();
    const sel = async (tag, name) => {
      const card = page.locator(tag, { hasText: name }).first();
      const cb = card.locator('.select-checkbox').first();
      if ((await cb.getAttribute('aria-checked')) !== 'true') await cb.click();
      await page.waitForTimeout(300);
    };
    await sel('vt-dataset-card', 'syn-imgs');
    await sel('vt-detector-card', 'doc-demo');
    await page.getByRole('button', { name: 'Train', exact: true }).click();
    await page.waitForSelector('.panel-center, vt-center-panel', { timeout: 30000 });
    await page.waitForTimeout(2000);
    await page.locator('.left-tab', { hasText: 'Manual' }).click();
    await page.waitForTimeout(600);
    const voteItem = async (idx, kind) => {
      await page.locator('.thumbnail-wrap:visible').nth(idx).click();
      await page.waitForSelector('.btn-good', { timeout: 8000 }).catch(() => {});
      await page.waitForTimeout(600);
      await page.locator(kind === 'good' ? '.btn-good' : '.btn-bad').first().click();
      await page.waitForTimeout(1100);
    };
    for (const i of [0, 1, 2, 3, 4]) await voteItem(i, 'good');
    for (const i of [5, 6, 7, 8]) await voteItem(i, 'bad');
    log('trained doc-demo (5 good / 4 bad)');
  }

  // 3. syn-patch (DINOv2 patch) for region voting
  await goDash();
  if (await hasDataset('syn-patch')) log('syn-patch exists, skipping');
  else await importSynthetic('syn-patch', 24, 'DINOv2 patch (region-aware images)');

  log('fixtures ready');
} finally {
  await browser.close();
}
