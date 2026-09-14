# Build slide decks from decks/*.deck manifests.
#
#   make                      every deck -> _out/*.pdf
#   make hold-the-line        one deck   -> _out/hold-the-line.pdf
#   make speaker              every deck -> _out/*.speaker.pdf (notes visible)
#   make unnumbered           every deck -> _out/*.unnumbered.pdf (no page numbers)
#   make editable             every deck -> _out/*.editable.pptx (real shapes; needs LibreOffice)
#   make pngs                 every deck -> _out/*-pngs*.zip (one image per page)
#   make html FMT=html        every deck -> _out/*.html
#   make watch DECK=hold-the-line   live preview in a browser
#   make check                preflight every manifest, build nothing
#   make list                 decks, slide counts, unused fragments

SHELL := /bin/bash
FMT   ?= pdf
DECKS := $(basename $(notdir $(wildcard decks/*.deck)))
DEPS  := $(wildcard fragments/*.md) $(wildcard figs/*) themes/vtsearch.css build.py

# render.sh is the ONLY place that invokes Marp. It carries two fixes a bare
# Marp call does not have -- `--no-stdin` (without it Marp waits for EOF on
# stdin, so a render started from a script or an agent shell hangs forever)
# and a PIPESTATUS check (without it a Marp that died still reports success,
# #3301). This Makefile used to call Marp itself and had neither, so building
# through `make` reproduced exactly the hang render.sh was written to prevent
# (#3434). Every target below therefore delegates.

.PHONY: all speaker unnumbered editable pngs check list clean watch $(DECKS)

all: $(addprefix _out/,$(addsuffix .$(FMT),$(DECKS)))

# Speaker view: each page is a miniature of the rendered slide beside its
# presenter notes. Two-pass (audience slides -> PNGs -> speaker deck).
speaker:
	@for d in $(DECKS); do ./render.sh $$d $(FMT) --speaker || exit 1; done

# The handover cut: no page numbers, written beside the numbered deck rather
# than over it.
unnumbered:
	@for d in $(DECKS); do ./render.sh $$d $(FMT) --no-pageno || exit 1; done

# Real PowerPoint shapes rather than one image per slide. Needs LibreOffice;
# see README.md for what the import does and does not survive.
editable:
	@for d in $(DECKS); do ./render.sh $$d pptx --editable || exit 1; done

# One image per page, zipped, for dropping the slides into another template as
# pictures. Unnumbered, because the receiving deck does its own numbering.
pngs:
	@for d in $(DECKS); do ./render.sh $$d png --no-pageno || exit 1; done

# Bare deck name builds that one deck, e.g. `make hold-the-line`.
$(DECKS): %: _out/%.$(FMT)

# render.sh runs build.py itself, so the manifest/fragment/figure deps are
# declared here only to decide *whether* to re-render, not how.
_out/%.$(FMT): decks/%.deck $(DEPS)
	@./render.sh $* $(FMT)

check:
	@./build.py --check

list:
	@./build.py --list

# Live-reloading preview. Ctrl-C to stop.
watch:
	@test -n "$(DECK)" || { echo "usage: make watch DECK=<name>"; exit 1; }
	@./render.sh $(DECK) --watch

clean:
	rm -rf _build _out
