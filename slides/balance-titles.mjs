#!/usr/bin/env node
// Check (and fix) where a full-bleed slide's headline breaks.
//
//     cd slides
//     ./build.py hold-the-line
//     npx @marp-team/marp-cli@4 _build/hold-the-line.md --theme-set themes/ \
//         --allow-local-files --html -o _out/hold-the-line.html
//     node balance-titles.mjs _out/hold-the-line.html            # report
//     node balance-titles.mjs _out/hold-the-line.html --write    # apply
//
// `slides/STYLE.md` has the rule: a headline is at most two lines, and when it
// is two, the break goes where the halves come out closest to equal. Neither
// half of that can be decided from the markdown, because where a headline
// wraps is a fact about rendered pixels — so this measures it in the same
// browser Marp rasterises with, on the real deck, by trying every word break
// and reading the line boxes back.
//
// The browser's own wrap is never the answer. CSS breaks *greedily*: it fills
// the first line as far as it will go and drops the rest, which is the most
// lopsided split available rather than the least. `Read All About It` came out
// 278px over 24px that way — "Read All About" and then a single word the width
// of one glyph. So every headline that takes two lines carries an explicit
// `<br>`, and this is what decides where.
//
// It reports rather than guesses when a headline cannot be fixed by breaking:
// three lines at any break means the words are too long for the 300px column
// and the headline needs rewriting, which is a person's job.

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { launchChromium } from '../scripts/screenshots/launch.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FRAGMENTS = path.join(HERE, 'fragments');

const html = process.argv[2];
const write = process.argv.includes('--write');
if (!html) {
  console.error('usage: balance-titles.mjs <deck.html> [--write]');
  process.exit(2);
}

/** Every `## ` headline in the fragment library, keyed by its flattened text. */
function fragmentHeadlines() {
  const byText = new Map();
  for (const name of fs.readdirSync(FRAGMENTS).filter((f) => f.endsWith('.md'))) {
    const file = path.join(FRAGMENTS, name);
    const lines = fs.readFileSync(file, 'utf8').split('\n');
    const i = lines.findIndex((l) => l.startsWith('## '));
    if (i < 0) continue;
    const raw = lines[i].slice(3).trim();
    byText.set(raw.replace(/<br\s*\/?>/g, ' ').replace(/\s+/g, ' '), { file, line: i, raw });
  }
  return byText;
}

const browser = await launchChromium();
const page = await browser.newPage({ viewport: { width: 1600, height: 1000 } });
await page.goto('file://' + path.resolve(html));
await page.waitForTimeout(1200);

// One measurement pass over the deck: for each distinct full-bleed headline,
// try the unbroken text and every single-word-boundary break, and read the
// resulting line boxes. Mutating the live h2 is what makes this faithful —
// same font, same size, same 300px column as the render.
const measured = await page.evaluate(() => {
  const widthsOf = (h2, scale) => {
    const range = document.createRange();
    range.selectNodeContents(h2);
    return [...range.getClientRects()]
      .map((b) => Math.round(b.width / scale))
      .filter((w) => w > 0);
  };
  const out = [];
  const seen = new Set();
  for (const sec of document.querySelectorAll('section.full')) {
    const h2 = sec.querySelector('h2');
    if (!h2) continue;
    // Read through the markup, not `textContent`: an existing `<br>` is an
    // element, so `textContent` glues the words either side of it into
    // "AboveAverage" and every split after that is measured on a word that
    // does not exist.
    const decoder = document.createElement('textarea');
    decoder.innerHTML = h2.innerHTML.replace(/<br\s*\/?>/gi, ' ').replace(/<[^>]+>/g, '');
    const flat = decoder.value.trim().replace(/\s+/g, ' ');
    if (seen.has(flat)) continue;
    seen.add(flat);
    const scale = sec.getBoundingClientRect().width / 1280;
    const original = h2.innerHTML;
    const words = flat.split(' ');
    const candidates = [];
    candidates.push({ at: 0, lines: ((h2.innerHTML = flat), widthsOf(h2, scale)) });
    for (let at = 1; at < words.length; at++) {
      h2.innerHTML = words.slice(0, at).join(' ') + '<br>' + words.slice(at).join(' ');
      candidates.push({ at, lines: widthsOf(h2, scale) });
    }
    h2.innerHTML = original;
    out.push({ flat, words, current: widthsOf(h2, scale), candidates });
  }
  return out;
});
await browser.close();

const byText = fragmentHeadlines();
let problems = 0;
let changed = 0;

for (const row of measured) {
  const frag = byText.get(row.flat);
  const where = frag ? path.basename(frag.file) : '(not in fragments/)';

  // The rule is a *maximum* of two lines, not a mandate of two, so breaking a
  // headline that would fit on one line stays the author's call — `Above` over
  // `Average` is a choice about the slide, not a wrap. What is never the
  // author's call is *where* a two-line headline breaks.
  const hasBreak = frag ? /<br\s*\/?>/i.test(frag.raw) : false;
  const oneLine = row.candidates.find((c) => c.at === 0 && c.lines.length === 1);
  const twoLine = row.candidates
    .filter((c) => c.at > 0 && c.lines.length === 2)
    .sort((a, b) => Math.abs(a.lines[0] - a.lines[1]) - Math.abs(b.lines[0] - b.lines[1]))[0];

  let want;
  if (oneLine && !hasBreak) want = { at: 0, lines: oneLine.lines };
  else if (twoLine) want = twoLine;
  else if (oneLine) want = { at: 0, lines: oneLine.lines };
  else {
    problems++;
    console.log(`REWRITE  ${where}: "${row.flat}" takes three lines at every break`);
    continue;
  }

  const wanted =
    want.at === 0
      ? row.flat
      : row.words.slice(0, want.at).join(' ') + '<br>' + row.words.slice(want.at).join(' ');
  const have = frag ? frag.raw : null;
  const shape = want.lines.join(' / ') + 'px';

  if (have === wanted) continue;
  problems++;
  // Two different findings, and conflating them makes the report unreadable.
  // PIN: the browser's greedy wrap already lands on the best break, and the
  // markdown is being made to say so, so a later edit cannot move it silently.
  // BREAK: the break actually moves, and the slide changes.
  const same = row.current.join() === want.lines.join();
  console.log(`${same ? 'PIN  ' : 'BREAK'} ${write ? '(fixed)' : '       '} ${where}: ${JSON.stringify(wanted)}`);
  console.log(`         now ${row.current.join(' / ')}px, best ${shape}`);
  if (write && frag) {
    const lines = fs.readFileSync(frag.file, 'utf8').split('\n');
    lines[frag.line] = '## ' + wanted;
    fs.writeFileSync(frag.file, lines.join('\n'));
    changed++;
  }
}

console.log(
  problems === 0
    ? `titles OK: ${measured.length} headlines, every one at most two lines and broken to balance`
    : `${problems} headline(s) want attention${write ? `, ${changed} rewritten` : ''}`,
);
process.exit(problems && !write ? 1 : 0);
