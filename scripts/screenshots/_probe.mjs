import { launchChromium } from './launch.mjs';
const OUT = process.env.OUT;
const browser = await launchChromium({ args: ['--no-sandbox', '--no-proxy-server'] });
const ctx = await browser.newContext({ viewport: { width: 1400, height: 1500 }, deviceScaleFactor: 2 });
const page = await ctx.newPage();
page.on('pageerror', e => console.log('[pageerror]', e.message));
page.on('console', m => { if (m.type()==='error') console.log('[console]', m.text()); });
await page.goto('http://localhost:5000', { waitUntil: 'domcontentloaded' });
await page.waitForTimeout(2000);
await page.click('button[aria-label="Import a new dataset"]');
await page.waitForSelector('.modal-content'); await page.waitForTimeout(1200);
async function shot(n){ const el = await page.$('.modal-content'); await el.screenshot({path:`${OUT}/${n}.png`}); console.log('shot',n); }
async function tab(l){ await page.click(`.importer-picker .tab-bar .tab:has-text("${l}")`); await page.waitForTimeout(1000); }
async function subtab(l){ await page.click(`.importer-subtab-bar .importer-subtab:has-text("${l}")`); await page.waitForTimeout(1400); }
await tab('Files'); await subtab('Folder'); await shot('a1-folder');
await page.click('.advanced-toggle'); await page.waitForTimeout(700);
await page.click('.media-type-trigger'); await page.waitForTimeout(400);
await page.click('.media-type-option:has-text("Image")'); await page.waitForTimeout(1200);
await shot('a2-folder-advanced-image');
await page.click('.advanced-toggle'); await page.waitForTimeout(500);
await page.click('.browse-row .btn'); await page.waitForTimeout(1600);
await shot('a3-folder-browse-open');
await page.click('.browse-row .btn'); await page.waitForTimeout(600);
await subtab('Manifest'); await shot('a4-manifest');
await tab('Demo'); await subtab('Downloaded Media'); await page.waitForTimeout(1200);
const row = await page.$('.demo-row'); if (row) { await row.click(); await page.waitForTimeout(1200); }
await shot('a5-demo-downloaded');
await page.click('.advanced-toggle'); await page.waitForTimeout(800);
await shot('a6-demo-advanced');
await browser.close();
