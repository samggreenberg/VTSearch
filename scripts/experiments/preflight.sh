#!/usr/bin/env bash
# Preflight for a GRID experiment launch.
#
# Every check here corresponds to a mistake that actually cost hours on a real
# study.  They are checks, not advice, because advice in a document did not stop
# any of them: the document existed and nobody read it.  Run this immediately
# before submitting arms.
#
#   bash scripts/experiments/preflight.sh --exp /exp/$USER/my-study --arms a,b,c
#
# Exits non-zero if anything is wrong.  `--warn-only` downgrades failures to
# warnings for the cases where you genuinely mean it (resuming a partial run).
#
# See scripts/experiments/GRID-PLAYBOOK.md for the SLURM-resource side of this
# (memory sizing, QOS caps, chunking) and LESSONS.md for the incident log.
set -uo pipefail

EXP=""
ARMS=""
NEED_GB="${PREFLIGHT_NEED_GB:-5}"
WARN_ONLY=0
REPO="${VTS_REPO:-}"
REGION_ARM=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --exp) EXP="$2"; shift 2 ;;
    --arms) ARMS="$2"; shift 2 ;;
    --need-gb) NEED_GB="$2"; shift 2 ;;
    --require-region-voting) REGION_ARM="$2"; shift 2 ;;
    --warn-only) WARN_ONLY=1; shift ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$EXP" ]] || {
  echo "usage: preflight.sh --exp DIR [--arms a,b,c] [--need-gb N]" >&2
  echo "                    [--require-region-voting DATASET:EMBEDDER]" >&2
  exit 2
}

FAILED=0
say_fail() {
  if [[ "$WARN_ONLY" == "1" ]]; then echo "  WARN  $*"; else echo "  FAIL  $*"; FAILED=1; fi
}
say_ok() { echo "  ok    $*"; }

echo "preflight: $EXP"

# --- 1. One experiment, one results dir -------------------------------------
# Two grids pointed at the same CALIB_EXP once.  The resume logic read the other
# grid's cells as "this arm is already complete" and aborted a whole overnight
# batch; had it not aborted, two different grids would have been silently mixed
# in one directory and analysed as one.
#
# Arm roots differ by study: the A/B launchers put arms under `results-ab/`, the
# acquisition and anchor sweeps under `results/`.  Checking only the first meant
# this check silently passed — did nothing at all — for every study of the second
# shape, which is the worse failure of the two: a gate that reports "ok" without
# having looked.
if [[ -n "$ARMS" ]]; then
  for arm in ${ARMS//,/ }; do
    seen=0
    for root in results-ab results; do
      cells="$EXP/$root/$arm/cells"
      [[ -d "$cells" ]] || continue
      seen=1
      n=$(find "$cells" -name 'task_*.csv' ! -name '*sweep*' 2>/dev/null | wc -l)
      if [[ "$n" -gt 0 ]]; then
        say_fail "arm '$arm' already has $n cell files in $cells"
        echo "        -> a fresh study needs its own --exp dir; a resume should pass --warn-only"
      else
        say_ok "arm '$arm' results dir is empty ($root/)"
      fi
    done
    [[ "$seen" == "1" ]] || say_ok "arm '$arm' results dir is new"
  done
fi

# --- 2. Free space on the REAL mount ----------------------------------------
# `df -h /exp` reported 394G free while the actual home, /exp/$USER, was its own
# 50G mount at 100%.  Cells died mid-write for hours on that misread.
# A fresh study's dir does not exist yet - that is the normal case - so stat the
# nearest existing ancestor, which is on the same filesystem.
STAT_PATH="$EXP"
while [[ ! -e "$STAT_PATH" && "$STAT_PATH" != "/" ]]; do
  STAT_PATH=$(dirname "$STAT_PATH")
done
MOUNT=$(df -P "$STAT_PATH" 2>/dev/null | awk 'NR==2 {print $6}')
AVAIL_GB=$(df -PBG "$STAT_PATH" 2>/dev/null | awk 'NR==2 {gsub(/G/,"",$4); print $4}')
if [[ -z "$AVAIL_GB" ]]; then
  say_fail "could not stat $STAT_PATH"
else
  if [[ "$AVAIL_GB" -lt "$NEED_GB" ]]; then
    say_fail "only ${AVAIL_GB}G free on $MOUNT (want >= ${NEED_GB}G)"
  else
    say_ok "${AVAIL_GB}G free on $MOUNT (the mount that actually holds $EXP)"
    # The mistake this guards against: reading a *parent* mount's free space.
    # /exp showed 394G while /exp/$USER was its own 50G mount at 100%.
    parent_mount=$(df -P "$(dirname "$STAT_PATH")" 2>/dev/null | awk 'NR==2 {print $6}')
    if [[ -n "$parent_mount" && "$parent_mount" != "$MOUNT" ]]; then
      echo "        note: $(dirname "$STAT_PATH") is a DIFFERENT mount ($parent_mount) - its free space is irrelevant"
    fi
  fi
fi

# --- 3. Zero-byte cells from a previous incident ----------------------------
# A cell killed mid-write leaves a 0-byte CSV.  It counts as "present" to the
# resume logic, so it is never re-run, and it crashes or silently shrinks the
# analysis later.
ZROOTS=()
for root in results-ab results; do [[ -d "$EXP/$root" ]] && ZROOTS+=("$EXP/$root"); done
if [[ "${#ZROOTS[@]}" -gt 0 ]]; then
  z=$(find "${ZROOTS[@]}" -name 'task_*.csv' ! -name '*sweep*' -size 0 2>/dev/null | wc -l)
  if [[ "$z" -gt 0 ]]; then
    say_fail "$z zero-byte cell files present - delete them or they will never be re-run"
  else
    say_ok "no zero-byte cell files"
  fi
fi

# --- 4. The worktree the jobs will actually import ---------------------------
# common.setup_env() puts VTS_REPO at the front of sys.path.  Unset, jobs
# silently import a different (stale) checkout.
if [[ -z "$REPO" ]]; then
  say_fail "VTS_REPO is unset - jobs will import whatever checkout the default points at"
elif [[ ! -d "$REPO/.git" && ! -f "$REPO/.git" ]]; then
  say_fail "VTS_REPO=$REPO is not a git worktree"
else
  say_ok "VTS_REPO=$REPO"
  if git -C "$REPO" rev-parse --abbrev-ref HEAD >/dev/null 2>&1; then
    branch=$(git -C "$REPO" rev-parse --abbrev-ref HEAD)
    git -C "$REPO" fetch -q origin 2>/dev/null
    local_sha=$(git -C "$REPO" rev-parse HEAD)
    remote_sha=$(git -C "$REPO" rev-parse "origin/$branch" 2>/dev/null || echo "")
    if [[ -n "$remote_sha" && "$local_sha" != "$remote_sha" ]]; then
      say_fail "worktree is not at origin/$branch (local ${local_sha:0:8}, remote ${remote_sha:0:8})"
      echo "        -> the code you committed is not the code that will run"
    else
      say_ok "worktree matches origin/$branch"
    fi
    if [[ -n "$(git -C "$REPO" status --porcelain --untracked-files=no)" ]]; then
      say_fail "worktree has uncommitted tracked changes - the run would be unreproducible"
    fi
  fi
fi

# --- 5. …and the checkout `import vtscore` ACTUALLY resolves to --------------
# VTS_REPO being right is not the same as the import being right.  #2846 launched
# from a fresh worktree with a correct VTS_REPO and correct PYTHONPATH and still
# imported the shared vts-calib checkout, via the venv's editable-install finder
# (the `.shadow` shim that neutralises it is untracked, so a new worktree has
# none).  That run only noticed because the branch had *added* a symbol; a branch
# that merely changes behaviour would have produced a clean, plausible, wrong
# table.  So resolve the import the way a job does - through common.setup_env() -
# and check where it landed.
if [[ -n "$REPO" && -d "$REPO/vtscore" ]]; then
  RESOLVED=$(CALIB_EXP="$EXP" python - "$REPO" <<'PY' 2>/dev/null
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(sys.argv[1]) / "scripts" / "experiments" / "calibration"))
import common
common.setup_env()
import vtscore
print(pathlib.Path(vtscore.__file__).resolve())
PY
)
  if [[ -z "$RESOLVED" ]]; then
    say_fail "could not resolve 'import vtscore' - is the venv active (source gridenv.sh)?"
  else
    REPO_REAL=$(cd "$REPO" && pwd -P)
    case "$RESOLVED" in
      "$REPO_REAL"/*) say_ok "import vtscore -> $RESOLVED" ;;
      *)
        say_fail "import vtscore resolves to $RESOLVED"
        echo "        -> that is NOT $REPO_REAL; the jobs would measure another checkout"
        echo "        -> source this worktree's gridenv.sh (it creates .shadow and pins VTS_REPO)"
        ;;
    esac
  fi
fi

# --- 6. The environment's PREMISE, not the flag that requests it -------------
# #2877 ran a whole study on `visual_genome_m x siglip` believing it was region
# voting.  It was not: `region_voting=True` is a *request*, and the harness
# silently falls back to whole-image training, whole-image scoring and the
# binary blend schedule when the medias carry no `patch_grid`.  Nothing was
# broken, so nothing complained; a report, a PR and a headline recommendation
# had to be corrected.  A flag you passed is not a property you got.
#
# Opt-in, because most studies do not claim region voting — but any study whose
# *rationale* rests on the scoring geometry ("a max over region nodes") should
# pass it.  One pickle open, and it either holds or it does not.
if [[ -n "$REGION_ARM" ]]; then
  ds="${REGION_ARM%%:*}"; emb="${REGION_ARM##*:}"
  if [[ -z "$ds" || -z "$emb" || "$ds" == "$REGION_ARM" ]]; then
    say_fail "--require-region-voting wants DATASET:EMBEDDER, got '$REGION_ARM'"
  else
    VERDICT=$(CALIB_EXP="$EXP" python - "$REPO" "$ds" "$emb" <<'PY' 2>&1
import pathlib, sys
repo, ds, emb = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, str(pathlib.Path(repo) / "scripts" / "experiments" / "calibration"))
import common
common.setup_env()
from _cells_io import load_medias
from vtscore.datasets import loader as _loader

pkl = _loader.EMBEDDINGS_DIR / f"{ds}__{emb}.pkl"
if not pkl.exists():
    print(f"MISSING {pkl}")
    raise SystemExit(0)
medias = load_medias(pkl)
n = len(medias)
grid = sum(1 for m in medias.values() if m.get("patch_grid") is not None)
print(("HOLDS" if grid == n and n else "FAILS") + f" patch_grid={grid}/{n} {pkl}")
PY
)
    case "$VERDICT" in
      HOLDS*) say_ok "region-voting premise ${ds} x ${emb}: ${VERDICT#HOLDS }" ;;
      FAILS*)
        say_fail "region-voting premise ${ds} x ${emb} does NOT hold: ${VERDICT#FAILS }"
        echo "        -> region_voting=True would silently run BINARY voting here (see #2877)"
        ;;
      *) say_fail "could not check the region-voting premise: $VERDICT" ;;
    esac
  fi
fi

# --- 7. Patch STYLES require box supervision --------------------------------
# Check 6 asks whether the region geometry is present.  This asks whether it is
# usable.  On a boxless dataset a Good vote has no box to pool, so it falls back
# to the image-level vector, while every Bad vote floods the full-image row plus
# ~197 raw patches as negatives.  No patch row is ever positive: the geometry can
# only teach "patch-like => negative", and max-pooling it at inference re-opens
# the asymmetry that produced perfect ranking, zero FPR and catastrophic FNR
# (see the module docstring in vtscore/eval/patch_styles.py).
#
# Reads the study's own config, so it needs no arguments and cannot drift from
# what the run will actually do.
if [[ -n "$REPO" && -f "$REPO/scripts/experiments/calibration/experiment_config.py" ]]; then
  STYLE_VERDICT=$(CALIB_EXP="$EXP" python - "$REPO" <<'PY' 2>&1
import pathlib, sys
repo = sys.argv[1]
sys.path.insert(0, str(pathlib.Path(repo) / "scripts" / "experiments" / "calibration"))
import experiment_config as cfg

if not hasattr(cfg, "styles_for"):
    print("SKIP config has no dataset-aware styles_for()")
    raise SystemExit(0)

bad = []
for ds in cfg.DATASETS:
    boxed = cfg.BOXED_BY_DATASET.get(ds, False)
    for emb in cfg.embedders_for_dataset(ds):
        styles = [st for st in cfg.styles_for(ds, emb) if st != "whole_image"]
        if styles and not boxed:
            bad.append(f"{ds}x{emb}={','.join(styles)}")
print(("FAILS " + "; ".join(bad)) if bad else "HOLDS")
PY
)
  case "$STYLE_VERDICT" in
    HOLDS*) say_ok "patch styles only on boxed datasets" ;;
    SKIP*)  say_ok "patch-style check skipped (${STYLE_VERDICT#SKIP })" ;;
    FAILS*)
      say_fail "patch styles on a BOXLESS dataset: ${STYLE_VERDICT#FAILS }"
      echo "        -> no Good vote can land on a patch row there; the patch rows are"
      echo "           negatives only, which is the boxless-max_patch failure mode"
      ;;
    *) say_fail "could not check patch styles: $STYLE_VERDICT" ;;
  esac
fi

echo
if [[ "$FAILED" == "1" ]]; then
  echo "PREFLIGHT FAILED - fix the above before launching (or --warn-only if deliberate)"
  exit 1
fi
echo "preflight OK"
