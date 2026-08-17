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
import { launchChromium } from './launch.mjs';

const APP = process.env.APP || 'http://localhost:5000';
const log = (...a) => console.log('[fixtures]', ...a);

const browser = await launchChromium();
try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  const goDash = async () => {
    await page.goto(`${APP}/#/dashboard`, { waitUntil: 'domcontentloaded' });
    // An empty registry renders the ".empty-state" placeholder instead of the
    // table — and this bootstrap's whole job is to fill an empty registry —
    // so wait for either, not just the table.
    await page.waitForSelector('.dash-table, .empty-state', { timeout: 30000 });
    await page.waitForTimeout(1200);
  };
  const hasDataset = (name) => page.locator('tr[vt-dataset-card]', { hasText: name }).count();
  const hasDetector = (name) => page.locator('tr[vt-detector-card]', { hasText: name }).count();

  const importSynthetic = async (name, size, embedderLabel) => {
    await page.locator('button[title="Import a new dataset"]').click();
    // The tab bar was renamed .importer-tab-bar -> .tab-bar (.tab) in the
    // header/layout IA unification; scope to .importer-picker to stay unique.
    await page.waitForSelector('.importer-picker .tab-bar', { timeout: 15000 });
    await page.locator('.importer-picker .tab', { hasText: 'Demo' }).click();
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
      const c = [...document.querySelectorAll('tr[vt-dataset-card]')].find((e) => e.textContent.includes(n));
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
    log('doc-demo exists, skipping create');
  } else {
    await page.locator('button[title="Create a new detector"]').click();
    await page.waitForTimeout(800);
    await page.locator('input[placeholder*="dog barking" i]').fill('colorful geometric pattern');
    await page.waitForTimeout(300);
    await page.locator('#detector-name').fill('doc-demo');
    await page.getByRole('button', { name: 'Create', exact: true }).click();
    await page.waitForTimeout(2000);
    log('created doc-demo');
  }
  {
    // Train it: enforce a baseline of exactly 5 good / 4 bad votes (and no
    // others) through the votes API - even when doc-demo already existed.
    // Later shot recipes mutate the votes (find-view adds corrections), so
    // a rerun against a surviving doc-demo used to drift off the 9-vote
    // baseline the autopilot-progress shot depends on.
    const ctx = await page.evaluate(async () => {
      const ds = await (await fetch('/api/datasets/registry')).json();
      const det = await (await fetch('/api/detectors/registry')).json();
      return {
        dataset: ((ds.datasets || []).find((x) => x.name === 'syn-imgs') || {}).id,
        detector: ((det.detectors || []).find((x) => x.name === 'doc-demo') || {}).id,
      };
    });
    if (!ctx.dataset || !ctx.detector) throw new Error('fixture context missing: ' + JSON.stringify(ctx));
    await page.evaluate(async ({ dataset, detector }) => {
      const h = { 'content-type': 'application/json', 'X-Dataset-Id': dataset, 'X-Detector-Id': detector };
      // Make sure both contexts are actually loaded server-side; right after
      // the Create flow the detector load can still be settling and votes
      // 409 with detector_not_loaded. NB: the load requests must NOT carry
      // X-Detector-Id - resolving the header for a not-yet-loaded detector
      // 409s the load call itself.
      const hLoad = { 'content-type': 'application/json', 'X-Dataset-Id': dataset };
      await fetch('/api/datasets/registry/' + dataset + '/load', { method: 'POST', headers: hLoad });
      await fetch('/api/detectors/registry/load', { method: 'POST', headers: hLoad, body: JSON.stringify({ detector_id: detector }) });
      const until = Date.now() + 30000;
      while (Date.now() < until) {
        const s = await (await fetch('/api/dataset/status', { headers: h })).json();
        if (s.loaded) break;
        await new Promise((r) => setTimeout(r, 500));
      }
      const items = await (await fetch('/api/medias/ids', { headers: h })).json();
      const ids = items.map((m) => m.id).sort((a, b) => a - b);
      const vote = async (id, target) => {
        for (let attempt = 0; attempt < 20; attempt++) {
          const r = await fetch('/api/medias/' + id + '/vote', { method: 'POST', headers: h, body: JSON.stringify({ target }) });
          if (r.ok) return;
          if (r.status !== 409) throw new Error('vote ' + id + ' -> ' + r.status);
          await new Promise((res) => setTimeout(res, 1000));
        }
        throw new Error('vote ' + id + ' still 409 after retries');
      };
      for (const id of ids.slice(0, 5)) await vote(id, 'good');
      for (const id of ids.slice(5, 9)) await vote(id, 'bad');
      for (const id of ids.slice(9)) await vote(id, 'none');
    }, ctx);
    // Give the per-vote retrain a moment to settle before the next step.
    await page.waitForTimeout(3000);
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
