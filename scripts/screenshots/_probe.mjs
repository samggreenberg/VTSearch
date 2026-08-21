import { launchChromium } from './launch.mjs';
const OUT = process.env.OUT;
const BASE = 'http://localhost:5000';
const browser = await launchChromium({ args: ['--no-sandbox', '--no-proxy-server'] });
const ctx = await browser.newContext({ viewport: { width: 1400, height: 1100 }, deviceScaleFactor: 2 });
const page = await ctx.newPage();
await page.goto(BASE, { waitUntil: 'domcontentloaded' });
await page.waitForTimeout(2000);
await page.click('button[aria-label="Import a new dataset"]');
await page.waitForSelector('.modal-content', { timeout: 15000 });
await page.waitForTimeout(1200);

async function shot(name) {
  const el = await page.$('.modal-content');
  await el.screenshot({ path: `${OUT}/${name}.png` });
  console.log('shot', name);
}
async function tab(label) { await page.click(`.importer-picker .tab-bar .tab:has-text("${label}")`); await page.waitForTimeout(1000); }
async function subtab(label) { await page.click(`.importer-subtab-bar .importer-subtab:has-text("${label}")`); await page.waitForTimeout(1500); }

await tab('Services'); await shot('01-services');
const svcSubs = await page.$$eval('.importer-subtab-bar .importer-subtab', els => els.map(e => e.textContent.trim())).catch(()=>[]);
console.log('svc subs', svcSubs);
await tab('Files'); await subtab('Folder'); await shot('02-files-folder');
await subtab('Manifest'); await shot('03-files-manifest');
await tab('Demo'); await subtab('Downloaded Media'); await shot('04-demo-downloaded');
// select first ready demo row
const row = await page.$('.demo-row');
if (row) { await row.click(); await page.waitForTimeout(1200); await shot('05-demo-row-selected'); }
await subtab('Synthetic Media'); await page.waitForTimeout(1200); await shot('06-demo-synth');
const row2 = await page.$('.demo-row');
if (row2) { await row2.click(); await page.waitForTimeout(1200); await shot('07-demo-synth-selected'); }
await browser.close();
