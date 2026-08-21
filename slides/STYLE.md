# Deck house rules

These apply to **every** deck in `slides/`, whatever the topic or audience.
They are not calibration-talk preferences; they are what a VTSearch deck looks
like. `slides/README.md` covers the mechanics (manifests, fragments, building);
this file covers the choices.

Where a rule is machine-checkable it is checked, and the check is named below.
Where it isn't, it is still a rule.

<!-- item-sep -->

## No running footer

A footer that reads *"VTSearch · calibration"* on all seventeen slides tells the
room something they worked out from the title slide and have not forgotten
since. It costs a line of the layout on every slide to say nothing.

So: no `footer:` key in a `.deck` manifest, and the theme styles no footer
element. The page number stays — that one is genuinely used, when someone asks
a question about "the slide with the mixture plot".

The identity of the talk belongs on the **title slide**, once.

<!-- item-sep -->

## Do not invent an affiliation

Never put an organisation, lab, programme, or sponsor name on a slide unless
you can point at where in this repo it came from. A plausible-looking
affiliation copied from some other deck is worse than no affiliation: it is
wrong in a way that only the audience will notice.

Venue and date on the title slide are fine when they are real.

<!-- item-sep -->

## Subscripts are subscripts

`w_f` is a variable named *w* with a subscript *f*. Written as three ASCII
characters at 34px it reads as a typo. Author it as HTML — Marp passes it
through, and the theme sizes and shifts it in the deck's own typeface:

```markdown
## cost = *w*<sub>f</sub>·FPR + *w*<sub>n</sub>·FNR
```

`<sup>` likewise. Do **not** reach for `$…$` math: Marp will render it, but
MathJax sets it in Computer Modern, and a serif formula in the middle of a
Helvetica headline looks like it was pasted in from a different document.

An underscore is fine inside a backticked identifier — `rule_inefficiency` is
the literal name of a column, not a subscript, and code type says so.

<!-- item-sep -->

## Colour means something; emphasis does not use colour

Two halves of one rule.

**Emphasis is weight, size, and darkness.** Bold text is *darker and heavier*.
It is never blue. A word that changes hue is making a claim — the audience has
to work out what the colour is telling them — and "this bit is important" does
not deserve a colour when bold already says it.

**Hue is reserved for identity, and the identity is shared with the figure
beside it.** `themes/vtsearch.css` defines three, matching the palette in
`slides/figs/src/make-calib-figs.py`:

| class | hue | means |
|---|---|---|
| `.cut` | blue | the threshold, and the shipped decision it makes |
| `.neg` | rust | the negative side: the Bad component, the losing arm |
| `.pos` | green | the positive side: the Good component, a measured win |

Use them only when the coloured word names something the figure on that slide
also draws in that colour:

```markdown
- Cut at the <span class="cut">midpoint</span> between the
  <span class="neg">low</span> and <span class="pos">high</span> modes
```

If the slide has no figure, it has no reason to colour a word.

<!-- item-sep -->

## The type floor: 20px at 1280×720

Nothing on a slide renders below **20px** — which is the page number, the
smallest thing the theme draws. Body copy is 28px.

This binds figures too, and figures are where it gets broken, because the size
a figure's labels *render* at is not the size they were *set* at. A figure `W`
inches wide is drawn at `W × 72` points and displayed in a slot `P` pixels
wide, so it renders at `P / (W × 72)` pixels per point. A 12.8in report figure
in a 717px sidebar renders its 10pt tick labels at **8 pixels**. It looked fine
in the report and it is unreadable from the third row.

Two consequences for any figure headed for a slide:

- **Size it to the slot, not to the page.** The standard sidebar
  (`![bg right:56% fit]`) is a 717×720 box — very nearly *square*. A 2:1 report
  figure fills half of it and wastes the rest, which is the same decision as
  drawing everything at half size. Six panels go 3 rows × 2 cols, not 2 × 3.
- **Then subtract.** One shared legend, not one per panel. No suptitle — the
  slide's own headline is the title. No footnote — that is a presenter note.

`slides/figs/src/slide_figure.py` holds the constants and `save()`, which
refuses to write a figure whose smallest label misses the floor. Use it for
every generated figure; a hand-checked floor is a floor that drifts.

Screenshots are exempt from the check and not from the rule: if the UI text in
a screenshot is unreadable at slot size, crop it or zoom the app, do not shrink
the caption to match.

<!-- item-sep -->

## Every deck opens with an outline

After the title slide, before the first argument: one slide naming the sections
in the order they arrive. Not a table of contents of slide titles — the three
or five things the talk is *for*.

Give it `<!-- _class: outline -->`, write it as a numbered list of
`**Name** *— one qualifying clause*`, and make each name match the headline of
the section it points at. If the audience can predict every headline in the
deck from this slide, it is doing its job: the interest in a research talk is
in the evidence, not in the reveal.

An outline lives in its own fragment (`fragments/outline-<deck>.md`) so a deck
that re-tailors the argument gets its own, rather than inheriting a list that
no longer describes it.

<!-- item-sep -->

## Builds: design the final slide, then chop

A mechanism slide that the room should watch assemble is a **build** — a
series of pages sharing one page number (see `slides/README.md` for the
markers). Two rules keep a build honest:

- **The final page is the slide.** Design it complete, as if there were no
  build; every earlier page is that slide with later steps *removed*, never a
  different layout. Nothing may move, resize, or restyle between pages — a
  reveal adds ink, and that is all it does. (For generated figures,
  `slide_figure.tight_box` pins every stage to the final stage's crop for
  exactly this reason.)
- **Chop at the mechanism's own joints.** One reveal per step the speaker
  narrates, not per bullet and not per sentence. A build that advances on
  every line is a slow way to read a list; a build that reveals "and now the
  models swap halves" is a mechanism teaching itself.

The speaker build always shows the one complete page — a speaker glancing at
notes needs the whole picture, not whichever stage the audience is on.
