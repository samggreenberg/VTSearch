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

- **Size it to the slot, not to the page.** The slot is the full 1280×720
  slide (see *Size a figure to a 16:9 slot* above). A 1:1 figure fills 56% of
  it and wastes the rest, which is the same decision as drawing everything at
  three-quarter size. Six panels go 2 rows × 3 cols, not 3 × 2.
- **Then subtract.** One shared legend, not one per panel. No suptitle — the
  slide's own headline is the title. No footnote — that is a presenter note.

`slides/figs/src/slide_figure.py` holds the constants and `save()`, which
refuses to write a figure whose smallest label misses the floor. Use it for
every generated figure; a hand-checked floor is a floor that drifts.

Screenshots are exempt from the check and not from the rule: if the UI text in
a screenshot is unreadable at slot size, crop it or zoom the app, do not shrink
the caption to match.

<!-- item-sep -->

## A figure owns the whole slide; the title lives in a notch

A figure that shares its slide with a column of bullets is not paying for that
column in the way it looks like it is. Measure before you rearrange: every
sidebar figure in this repo was already **height-bound** in its slot — sized to
the slot, so it touched 720px top to bottom and left its slack out the sides.
The text column sat beside empty *figure margin*, not on top of the drawing.
Deleting the bullets bought those figures exactly nothing.

Two consequences, and the second is the one that surprises people:

- **Height is the scarce resource, not width.** A title band across the top of
  the slide spends the only dimension that binds. Measured over this deck's
  fifteen figures, a 1280×605 band made eleven of them **0.84×** the size they
  already were.
- **Full-bleed is never worse**, and up to **1.57×** better on a wide figure.
  So the standard is `<!-- _class: full -->` with `![bg fit]`, and the title
  goes in a **notch** cut out of the drawing's top-left corner.

`slide_figure.TITLE_NOTCH_PX` is that rectangle — 300×200 at a 60×42 inset —
and `section.full` in `themes/vtsearch.css` is the same rectangle in CSS. Keep
the two in step. `save()` refuses to write a full-bleed figure that draws
inside it, so the reserve cannot rot.

**The height is a measurement, not a round number.** It is the deck's longest
full-bleed headline plus a little: every `section.full h2` renders at or under
191.2px in a 300px column, so the reserve is 200. It used to be 250, sized
against the sentence-length headlines the figure slides carried before #3242
retitled them to short phrases, and the 59px nobody used were 59px taken out of
every full-bleed figure. Re-measure before you change it, and re-measure if a
headline grows past two lines — render the deck to HTML and read the boxes:

```bash
./build.py hold-the-line
npx @marp-team/marp-cli@4 _build/hold-the-line.md --theme-set themes/ \
    --allow-local-files --html -o _out/hold-the-line.html
# then, in a browser: for every `section.full h2`, its height / (slide width / 1280)
```

**The notch does not move, and that is the whole point of it.** A headline that
shifts corner to corner to dodge each figure stops being a headline and becomes
another thing to hunt for. So a figure that genuinely cannot spare its top-left
corner **carries no title at all** — it does not put one somewhere else. That
is the standard working, not failing.

**Its height is the one part a figure may trim**, because 200px is a reserve
for the deck's *longest* headline and most slides carry a shorter one. Pass the
rectangle to `save(notch=...)`, with the height measured — by the recipe above,
on the slide the figure actually appears on — rather than guessed, and re-take
that measurement if the headline changes. `vote-boundary` is the one figure
that does it: "Rock the Vote" is one line and measures 56.8px, and the 100px of
unused reserve left a band under the title with no title in it and no items
either. Do not trim x, y or width; the notch's *position* is the standard, and
a figure whose ink reaches the top-left corner still carries no title.

Three rules follow.

**No kicker on a full-bleed slide.** The figure already says which mechanism
this is; `### Iteration 1 — the idea` over the top of it is a second thing to
read before the first. Say it instead — it belongs in the presenter notes.

**The headline is meant to wrap.** 40px in a 300px column, three or four lines
deep. That is not a compromise to fit the notch, it is what makes the notch
narrow enough to be clearable — see below — and it buys a bigger headline than
the deck's own 34px `h2`.

**Which figures can clear it is geometry, not taste.** A schematic drawn
symmetrically about a spine puts its first row — the block, plus any labels
hanging off its left edge — at a fixed fraction of its own width, near 0.29,
*whatever* its aspect: widening the drawing moves the block and the notch
together. So a notch wider than that is unclearable at any aspect a slide can
show, which is why 300px and not 420px.

The hard case is a figure whose **top row spans the drawing** — a score axis or
a scatter that starts in the top-left by construction. This used to be where a
slide gave up its title. It is not, and #3242 is the proof: every full-bleed
figure in the deck now clears the reserve, and `save()` enforces it on all of
them rather than taking a `notch=False`. The blocker in that case is
*horizontal*, so shortening the reserve moves nothing out of it — which is why
the repair is one of these three, in rising order of cost:

- **Pan the frame.** A height-limited figure keeps its scale when its canvas
  grows to the left, so the drawing rides right without shrinking. It buys half
  a unit of clearance per unit of gutter, because the wider canvas re-centres.
  `calib-quantile-flow` and `vote-boundary` are repaired this way and pay
  nothing at all.
- **Move the one thing that reaches left.** Often the ink in the reserve is a
  single object, not the drawing: `calib-acq-flow`'s `D_0` block, or the
  "Unlabeled" label that used to hang off the left edge of `D`<sub>−1</sub> in
  three figures and held all three two units right of where they wanted to sit.
  Drop it, shift it to the other side, or give its row back the space
  elsewhere.
- **Indent the axis.** When the top row genuinely spans the drawing —
  `calib-knob-flow`, `-walk-`, `-tilt-` — the panel starts right of the notch
  and spends the right margin to buy most of the width back. Those three give
  up 16% of their span, which is what a headline costs there.

A figure that resists all three still carries no title, and its headline
becomes the first line of the notes. That is the standard working, not
failing — but check the three first, because none of them was tried before.

<!-- item-sep -->

## Size a figure to a 16:9 slot, not to a square one

The old rule here read "the standard sidebar is a 717×720 box — very nearly
*square*, so six panels go 3 rows × 2 cols, not 2 × 3." That was right for the
sidebar and is exactly wrong now: a full-bleed slot is **1280×720**, and a tall
grid in a wide box is the same decision as drawing everything at half size.
Six panels go **2 rows × 3 cols**.

The arithmetic behind the type floor is unchanged — a figure `W` inches wide is
drawn at `W × 72` points and displayed in a `P`-pixel slot, so it renders at
`P / (W × 72)` pixels per point — but the binding axis has moved. A drawing
narrower than 16:9 is **height**-bound, and adding width to it buys nothing at
all: the empty side margins grow and the type stays exactly the size it was.
The only thing that makes a height-bound figure bigger is making it **shorter**
— moving rows sideways, not stretching them.

So when a schematic goes full-bleed, look for the height:

- **Long diagonals are where it hides.** The loop schematic's fork was two long
  diagonal arrows costing three units of drop; on a wide canvas the same two
  outputs sit low and far apart and the fork is nearly horizontal.
- **A wide panel wants the width.** The Part 2 figures hang everything on one
  score axis. Widening that axis spends the new width on the thing the audience
  is actually asked to read, and costs no height.
- **Some figures cannot be reshaped by a constant.** Where a layout is *solved*
  from geometric constraints rather than set — `calib-quantile-flow`'s three
  equal panels, `vote-boundary`'s equal-aspect scatter — widening the canvas
  only adds empty space. Those need a real redesign, and until they get one
  they letterbox. Say so rather than stretching them.

<!-- item-sep -->

## A label is closer to its object than objects are to each other

A schematic — boxes, arrows, a thing labelled `D`<sub>1</sub> — is read by
proximity before it is read at all. The eye binds each label to whatever is
nearest, so the *only* thing that says which box `D`<sub>1</sub> names is that
it sits closer to that box than to anything else. Get that backwards, as
`calib-xcal-flow` did (issue #3217), and the reader has to work out from the
content what the layout should have told them for free.

So two gaps, and they are not close together:

| gap | what it separates | value |
|---|---|---|
| **label gap** | a label and the thing it labels | `slide_figure.LABEL_GAP_PT` |
| **object gap** | two distinct objects — a box and the arrow leaving it, an arrowhead and what it points at, two stacked lines of a conclusion | `slide_figure.OBJECT_GAP_PT` |

They are in **printed points**, not a figure's drawing units, so the standard
survives being dropped into any figure: divide by the points-per-unit that
figure is drawn at and use the result for every gap in it. The ratio is the
part that matters, and it wants to be roughly 3:1 — eyeballing the two
separately is what produces 1:1, which is exactly the failure above.

**An arrow is never shorter than the word written along it.** A block arrow
carries its label centred on the whole arrow, head included, so an arrow that
is shorter than the label plus two arrowheads prints its own word over its own
point. `make-calib-figs.arrow_len_for` measures the glyphs and returns the
minimum, `_labeled_arrow` asserts it, and a layout that wants a longer arrow
pays for it in canvas rather than in legibility — the canvas grows at the top,
where these schematics pin their first row, so no other row moves.

Two consequences worth spelling out, because they are the ones that get
fudged:

- **Measure the object gap from where the arrow's own line leaves the box**,
  not from the box's nearest flat edge. Otherwise a diagonal arrow is short of
  the box by a different amount than a vertical one, and one constant stops
  holding. `make-calib-figs._box_edge` is the helper.
- **An arrow that points at a group stops an object gap from the group's
  topmost ink** — which is usually the group's own label, not the line or the
  box you were thinking of.

<!-- item-sep -->

## Every deck opens with an outline, and comes back to it

After the title slide, before the first argument: one slide naming the sections
in the order they arrive. Not a table of contents of slide titles — the three
or five things the talk is *for*.

Give it `<!-- _class: outline -->` and write it as a numbered list of **bare
names**. No qualifying clause, no gloss, nothing the presenter is going to say
anyway: the room reads this slide at a glance or not at all, and a line they
have to study is a line that has stopped being an outline. Make each name match
the section it points at closely enough that the audience can place a slide
without being told.

**Then show it again before every section**, with that section's line marked.
Use the same fragment, and let the manifest do the marking:

```
outline-hold-the-line +at3
```

The theme takes the marked line to full weight and quiets the rest, which is
the deck's one rule for emphasis — weight and darkness, never hue — used to say
*you are here*. It costs one slide per section and buys two things: the listener
is anchored in the argument, and a topic change gets an unmistakable signal to
wake up for.

**The opening list is unmarked, and the marked first section follows it.** The
room reads `1 … 5` all one weight — every section still ahead of them — and
only then does section 1 go bold and the rest go quiet. Those two pages are not
a stutter: the first is the shape of the talk and the second is the entry into
it, and running them together (opening straight on `+at1`) means the whole list
is never once shown as a list. So a deck's outline appears *N* + 1 times for
*N* sections: bare, then `+at1`, section 1, `+at2`, section 2, and so on.

**It is laid out like the deck's other slides**, not as its own kind of page:
the headline sits in the same top-left notch a full-bleed figure leaves for it,
and the list occupies the rectangle the figure would. The outline *is* that
slide's figure — it is the one thing the room is asked to look at — so it gets
the slot the deck gives figures, and the title does not move on the one slide
that comes back six times.

An outline lives in its own fragment (`fragments/outline-<deck>.md`) so a deck
that re-tailors the argument gets its own, rather than inheriting a list that
no longer describes it — and one fragment used *N* ways cannot drift the way *N*
copies would.

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

**Every page of a build carries a letter** after the shared page number — 5a,
5b, 5c — so a reveal can be named. Write the presenter notes against those
letters (`**c** — the same cut, loosened and tightened`) rather than against
"page 3 of the build": the letter is what the audience deck actually prints,
what a question from the room will use, and what the speaker build's contact
sheet labels each frame with. `build.py --check` requires every page of a
group to be named by some note — one note may cover two, but no frame may go
unmentioned.

The speaker build shows the one complete page large — a speaker glancing at
notes needs the whole picture, not whichever stage the audience is on — with
the whole group beneath it as a lettered contact sheet. That sheet is why a
note never has to *describe* the build: "this slide is a seven-page build" is
a sentence spent saying what a picture already says, in the one column that
has no room to spare.
