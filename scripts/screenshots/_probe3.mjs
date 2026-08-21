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
await page.click('.importer-picker .tab-bar .tab:has-text("Files")'); await page.waitForTimeout(900);
await page.click('.importer-subtab-bar .importer-subtab:has-text("Folder")'); await page.waitForTimeout(1400);
await page.click('.btn:has-text("Browse")'); await page.waitForTimeout(1500);
await shot('20-folder-browse-open');
await browser.close();
