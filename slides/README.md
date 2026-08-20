# VTSearch slide decks

Text-based slides, so git can actually version them. Slides live as individual
markdown fragments in `slides/fragments/`; a deck is a manifest in
`slides/decks/` that names fragments in order. Re-tailoring a talk for a new
audience means writing a new manifest, not copying a deck.

```
fragments/  one fragment per slide — the reusable library
decks/      *.deck manifests: front matter + an ordered list of fragments
figs/       committed PNG/SVG figures (figs/src/ holds their generators)
themes/     vtsearch.css — the Marp theme
_build/     assembled markdown (gitignored)
_out/       rendered decks (gitignored)
```

`build.py --check` runs as a `./run-tests.sh` gate, so a deck that names a
missing fragment or figure fails the suite rather than rotting quietly.

## Build

Needs node and python3. Nothing to install — `npx` fetches Marp on first run
(~1 min, once). Run from this directory:

```bash
./render.sh scale26-review          # -> _out/scale26-review.pdf
./render.sh sponsor-brief html      # or html / pptx
./build.py --check                  # preflight all manifests, build nothing
./build.py --list                   # decks, slide counts, unused fragments
```

There's also a `Makefile` (`make`, `make FMT=html`, `make watch DECK=…`) but
**`make` is not installed on the laptop** — `render.sh` is the working path
here. The Makefile is for the grid or wherever make exists.

For live preview while writing:

```bash
./build.py sponsor-brief
npx @marp-team/marp-cli@4 _build/sponsor-brief.md \
    --theme-set themes/ --allow-local-files -w --preview
```

If Marp can't find a browser (it drives one to rasterise), point it at one:
`CHROME_PATH=/path/to/chrome ./render.sh <deck>`.

## Writing a slide

Copy `slides/fragments/_template.md`. Keep it short — if a slide needs more
than ~25 words it wants to be two slides or a figure.

Layout is a background-image directive, which is why placement survives in
version control as a diffable line rather than a repacked binary:

```markdown
![bg right:56% fit](figs/positive-starvation.png)

### Finding 2
## Positives are the binding constraint

- After **150 votes**: a median of only **4–11** positives
```

Figure paths are written relative to `slides/`, not to the fragment — `build.py`
repoints them when it assembles into `_build/`.

`bg right:56%` puts the figure in the right 56% of the slide; `left:` mirrors
it; plain `![bg fit]` goes full-bleed. Classes `lead`, `statement`, `full`, and
`caveat` are defined in the theme and set per-slide with `<!-- _class: full -->`.

Any HTML comment that isn't a Marp directive becomes a **presenter note** —
visible in `--preview` and exported into PPTX/HTML notes, not on the slide.

**Never put a bare `---` in a fragment**: Marp reads it as a slide break and
would silently split one slide into two. `build.py` errors on it. Use `***`
for a horizontal rule.

## Variants

For a slide that needs an audience-specific cut, add a suffix rather than
branching the deck: `region-voting.md` and `region-voting.short.md` are both
fragments, and each manifest picks the one it wants. Comment a line out of a
manifest with `#` to park a slide without losing it.

## Figures

Commit them. That is deliberate: a re-rendered figure costs one ~150 KB blob,
and only when *that figure* changes — editing slide text touches nothing but a
few KB of markdown. This is the property a `.pptx` lacks, where every edit
rewrites the whole archive.

- **PNG for data-heavy plots.** An SVG scatter of 26 000 points is one
  `<circle>` per point — far larger than the PNG, and it diffs as a single
  enormous line anyway.
- **SVG for diagrams** with a few dozen shapes: smaller, and genuinely
  diffable.
- Iterate in the working tree and `--amend` while a figure is still ugly, so
  only the final render becomes permanent history.

A figure generated from code keeps that code beside it in `figs/src/`, so the
plot can be regenerated when the underlying numbers move —
`slides/figs/src/make-calib-figs.py` is the worked example. Generators may
import `vtscore` directly (they run from a checkout that has it), which is what
lets a mechanism figure plot the *shipped* estimator rather than a redrawing of
it. matplotlib is already a project dependency, so a normal checkout install
runs them; if a generator ever wants something more exotic, install it ad hoc
rather than adding it to `requirements/` for a figure's sake.

## Exporting to PowerPoint

`make FMT=pptx` produces one full-slide image per slide — fine for handing over
a read-only deck, not editable. Marp's `--pptx-editable` emits real shapes but
needs LibreOffice installed. If a co-presenter must edit slides in PowerPoint,
export once at the end and don't commit the result.
