import { launchChromium } from './launch.mjs';
const OUT = process.env.OUT;
const browser = await launchChromium({ args: ['--no-sandbox', '--no-proxy-server'] });
const ctx = await browser.newContext({ viewport: { width: 1400, height: 1400 }, deviceScaleFactor: 2 });
const page = await ctx.newPage();
await page.goto('http://localhost:5000', { waitUntil: 'domcontentloaded' });
await page.waitForTimeout(2000);
await page.click('button[aria-label="Import a new dataset"]');
await page.waitForSelector('.modal-content'); await page.waitForTimeout(1200);
async function shot(n){ const el = await page.$('.modal-content'); await el.screenshot({path:`${OUT}/${n}.png`}); console.log('shot',n); }
async function tab(l){ await page.click(`.importer-picker .tab-bar .tab:has-text("${l}")`); await page.waitForTimeout(1000); }
async function subtab(l){ await page.click(`.importer-subtab-bar .importer-subtab:has-text("${l}")`); await page.waitForTimeout(1400); }
await tab('Files'); await subtab('Folder');
await page.click('.advanced-toggle'); await page.waitForTimeout(800);
await shot('10-files-folder-advanced');
// image media type to get more advanced options
await page.click('.media-type-trigger'); await page.waitForTimeout(400);
await page.click('.media-type-option:has-text("Image")'); await page.waitForTimeout(1200);
await shot('11-files-folder-advanced-image');
await tab('Services'); await page.waitForTimeout(1000); await shot('12-services');
await browser.close();
