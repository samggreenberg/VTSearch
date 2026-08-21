import { launchChromium } from './launch.mjs';
const OUT = process.env.OUT;
const browser = await launchChromium({ args: ['--no-sandbox', '--no-proxy-server'] });
const ctx = await browser.newContext({ viewport: { width: 1400, height: 1500 }, deviceScaleFactor: 2 });
const page = await ctx.newPage();
await page.goto('http://localhost:5000', { waitUntil: 'domcontentloaded' });
await page.waitForTimeout(2000);
await page.click('button[aria-label="Import a new dataset"]');
await page.waitForSelector('.modal-content'); await page.waitForTimeout(1200);
async function shot(n){ const el = await page.$('.modal-content'); await el.screenshot({path:`${OUT}/${n}.png`}); console.log('shot',n); }
await page.click('.importer-picker .tab-bar .tab:has-text("Demo")'); await page.waitForTimeout(900);
await page.click('.importer-subtab-bar .importer-subtab:has-text("Downloaded")'); await page.waitForTimeout(1500);
const opts = await page.$$eval('.media-type-option', els => els.map(e=>e.textContent.trim())).catch(()=>[]);
await page.click('.media-type-trigger'); await page.waitForTimeout(400);
const opts2 = await page.$$eval('.media-type-option', els => els.map(e=>e.textContent.trim()));
console.log('mediatypes', JSON.stringify(opts2));
for (const t of ['Document','Video']) {
  if (!opts2.includes(t)) continue;
  await page.click(`.media-type-option:has-text("${t}")`); await page.waitForTimeout(1600);
  const row = await page.$('.demo-row'); if (row) { await row.click(); await page.waitForTimeout(1000); }
  await page.click('.advanced-toggle'); await page.waitForTimeout(700);
  await shot(`b-demo-${t.toLowerCase()}-advanced`);
  await page.click('.advanced-toggle'); await page.waitForTimeout(400);
  await page.click('.media-type-trigger'); await page.waitForTimeout(400);
}
await browser.close();
