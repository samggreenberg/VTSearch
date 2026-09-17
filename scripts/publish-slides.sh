#!/usr/bin/env bash
# Render every slide deck and publish the PDFs as assets on a rolling GitHub
# release, so the newest deck always lives at a stable, linkable URL:
#
#   https://github.com/samggreenberg/VTSearch/releases/download/slides-latest/hold-the-line.pdf
#
# Release assets are the one GitHub surface that stores a binary *against* the
# repository without putting it *in* the repository: the PDF is a build
# artifact (slides/_out/ is gitignored), its derivation stays fully tracked in
# slides/fragments/ + slides/decks/, and nobody has to clone and run Marp to
# read the current talk. Issue attachments get a fresh URL per upload, Actions
# artifacts expire and need a login, and Pages would put the bytes back on a
# branch -- none of them give you one durable link.
#
#   ./scripts/publish-slides.sh                  # every deck in slides/decks/
#   ./scripts/publish-slides.sh hold-the-line    # just this one
#   ./scripts/publish-slides.sh --dry-run        # render, print, upload nothing
#
# .github/workflows/publish-slides.yml runs this on every push to dev that
# touches slides/, which is what keeps the link current without anyone
# remembering to. Running it by hand needs `gh` authenticated with write access
# (a cloud session has neither, which is why the workflow exists).
set -euo pipefail

cd "$(dirname "$0")/.."

TAG=${SLIDES_RELEASE_TAG:-slides-latest}
dry_run=
allow_dirty=
decks=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tag) TAG=${2:?--tag needs a value}; shift 2 ;;
        --dry-run) dry_run=1; shift ;;
        --allow-dirty) allow_dirty=1; shift ;;
        -h|--help) sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        -*) echo "unknown argument: $1" >&2; exit 1 ;;
        *) decks+=("$1"); shift ;;
    esac
done

# Publishing every deck is what lets the script also retire assets for decks
# that no longer exist; a named subset must not, because it cannot tell a
# deleted deck from one it simply wasn't asked about.
publishing_all=
if [[ ${#decks[@]} -eq 0 ]]; then
    publishing_all=1
    for manifest in slides/decks/*.deck; do
        decks+=("$(basename "$manifest" .deck)")
    done
fi
if [[ ${#decks[@]} -eq 0 ]]; then
    echo "no decks found in slides/decks/" >&2
    exit 1
fi

# The release body names the commit the PDFs were built from, so a dirty tree
# would make the release lie about its own provenance.
if [[ -z $allow_dirty && -n $(git status --porcelain -- slides/) ]]; then
    echo "ERROR: slides/ has uncommitted changes, so the published PDFs would not match" >&2
    echo "       the commit the release names. Commit them, or pass --allow-dirty." >&2
    exit 1
fi

# Marp needs a browser and does not always find one on its own. render.sh takes
# whatever CHROME_PATH says, so resolve it here rather than in each caller --
# the GitHub runner, the cloud container (/opt/pw-browsers) and a laptop all
# keep chromium somewhere different.
if [[ -z ${CHROME_PATH:-} ]]; then
    for candidate in google-chrome google-chrome-stable chromium chromium-browser; do
        if path=$(command -v "$candidate" 2>/dev/null); then
            export CHROME_PATH=$path
            break
        fi
    done
    if [[ -z ${CHROME_PATH:-} && -x /opt/pw-browsers/chromium ]]; then
        export CHROME_PATH=/opt/pw-browsers/chromium
    fi
fi
[[ -n ${CHROME_PATH:-} ]] && echo "using browser: $CHROME_PATH"

assets=()
for deck in "${decks[@]}"; do
    [[ -f "slides/decks/$deck.deck" ]] || { echo "no such deck: slides/decks/$deck.deck" >&2; exit 1; }
    echo "== rendering $deck"
    ( cd slides && ./render.sh "$deck" pdf )
    ( cd slides && ./render.sh "$deck" pdf --speaker )
    assets+=("slides/_out/$deck.pdf" "slides/_out/$deck.speaker.pdf")
done

for asset in "${assets[@]}"; do
    [[ -s $asset ]] || { echo "ERROR: $asset is missing or empty after rendering" >&2; exit 1; }
done

sha=$(git rev-parse HEAD)
built=$(TZ=UTC date +%Y-%m-%dT%H:%M:%SZ)
notes=$(printf '%s\n' \
    "Rendered slide decks, republished automatically on every push to \`dev\` that touches \`slides/\`." \
    "" \
    "| | |" \
    "|---|---|" \
    "| Commit | \`$sha\` |" \
    "| Built (UTC) | $built |" \
    "| Decks | ${decks[*]} |" \
    "" \
    "\`<deck>.pdf\` is the audience cut; \`<deck>.speaker.pdf\` puts a miniature of each" \
    "slide beside its presenter notes. Both are build artifacts -- the sources they" \
    "come from live in \`slides/\`, and \`slides/README.md\` says how to render them yourself." \
    "" \
    "This release is a rolling artifact drop, not a version of VTSearch.")

if [[ -n $dry_run ]]; then
    echo
    echo "-- dry run; nothing uploaded --"
    echo "tag:    $TAG @ $sha"
    printf 'asset:  %s\n' "${assets[@]}"
    exit 0
fi

command -v gh >/dev/null 2>&1 || {
    echo "ERROR: gh is not installed. The PDFs are rendered in slides/_out/; publishing" >&2
    echo "       needs the GitHub CLI authenticated with write access to this repo." >&2
    exit 1
}

# Move the tag with each publish so the release points at the commit its assets
# were actually built from. GitHub will not re-target a release whose tag
# already exists, so the tag -- not `gh release edit --target` -- is what has to
# move; force-pushing a deliberately rolling tag is the intended use of one.
echo "== tagging $TAG -> $sha"
git push --force origin "$sha:refs/tags/$TAG"

if gh release view "$TAG" >/dev/null 2>&1; then
    echo "== updating release $TAG"
    gh release edit "$TAG" --notes "$notes"
else
    echo "== creating release $TAG"
    # --prerelease keeps a rolling deck drop out of the "Latest release" slot,
    # which belongs to a real version tag if VTSearch ever cuts one. The
    # /releases/download/<tag>/ URL this exists for is unaffected either way.
    gh release create "$TAG" --title "Slide decks (latest)" --notes "$notes" --prerelease
fi

echo "== uploading ${#assets[@]} asset(s)"
gh release upload "$TAG" "${assets[@]}" --clobber

# A deck that was deleted leaves its PDF behind otherwise, and a stale asset on
# a release called "latest" is worse than a missing one.
if [[ -n $publishing_all ]]; then
    expected=$(printf '%s\n' "${assets[@]}" | xargs -n1 basename | sort)
    while read -r stale; do
        [[ -n $stale ]] || continue
        echo "== removing stale asset $stale"
        gh release delete-asset "$TAG" "$stale" --yes
    done < <(gh release view "$TAG" --json assets --jq '.assets[].name' | sort | comm -23 - <(printf '%s\n' "$expected"))
fi

repo=$(gh repo view --json nameWithOwner --jq .nameWithOwner)
echo
echo "published:"
for asset in "${assets[@]}"; do
    echo "  https://github.com/$repo/releases/download/$TAG/$(basename "$asset")"
done
