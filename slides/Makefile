# Build slide decks from decks/*.deck manifests.
#
#   make                      every deck -> _out/*.pdf
#   make scale26-review       one deck   -> _out/scale26-review.pdf
#   make html FMT=html        every deck -> _out/*.html
#   make watch DECK=sponsor-brief   live preview in a browser
#   make check                preflight every manifest, build nothing
#   make list                 decks, slide counts, unused fragments

SHELL := /bin/bash
MARP  := npx --yes @marp-team/marp-cli@4
FMT   ?= pdf
DECKS := $(basename $(notdir $(wildcard decks/*.deck)))
FIGS  := $(wildcard figs/*)

.PHONY: all check list clean watch $(DECKS)

all: $(addprefix _out/,$(addsuffix .$(FMT),$(DECKS)))

# Bare deck name builds that one deck, e.g. `make sponsor-brief`.
$(DECKS): %: _out/%.$(FMT)

_build/%.md: decks/%.deck $(wildcard fragments/*.md) build.py
	@./build.py $*

# --allow-local-files is required for figs/ to appear in PDF and PPTX output.
# Marp only *warns* when a figure path doesn't resolve and still exits 0, which
# yields a silently figure-less deck. Promote that warning to a build failure.
_out/%.$(FMT): _build/%.md themes/vtsearch.css $(FIGS)
	@mkdir -p _out
	@set -o pipefail; $(MARP) $< --theme-set themes/ --allow-local-files -o $@ \
	  2>&1 | tee _out/.$*.log
	@if grep -q "local files are missing" _out/.$*.log; then \
	  rm -f $@ _out/.$*.log; \
	  echo "ERROR: Marp could not resolve some figures. Deck removed."; exit 1; \
	fi
	@rm -f _out/.$*.log

check:
	@./build.py --check

list:
	@./build.py --list

# Live-reloading preview. Ctrl-C to stop.
watch:
	@test -n "$(DECK)" || { echo "usage: make watch DECK=<name>"; exit 1; }
	@./build.py $(DECK)
	$(MARP) _build/$(DECK).md --theme-set themes/ --allow-local-files -w --preview

clean:
	rm -rf _build _out
