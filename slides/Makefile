# Build slide decks from decks/*.deck manifests.
#
#   make                      every deck -> _out/*.pdf
#   make hold-the-line        one deck   -> _out/hold-the-line.pdf
#   make speaker              every deck -> _out/*.speaker.pdf (notes visible)
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

.PHONY: all speaker check list clean watch $(DECKS)

all: $(addprefix _out/,$(addsuffix .$(FMT),$(DECKS)))

# Speaker view: each page is a miniature of the rendered slide beside its
# presenter notes. Two-pass (audience slides -> PNGs -> speaker deck).
speaker:
	@for d in $(DECKS); do ./render.sh $$d $(FMT) --speaker || exit 1; done

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
