# Achievements

VTSearch's achievements are a light gamification layer: ten usage milestones,
each with four tiers, a Total Score, an unlock toast and a dot on the trophy
button. What a user sees, and the table of every entry with its thresholds, is
the [User Guide's Achievements section](user/USER_GUIDE.md#achievements). This
page is how it works and how to add an entry. The slides after the
`hold-the-line` deck's Conclusion show the same system to a room
(`slides/fragments/achievement-*.md`).

The whole system is **app-tier**. The library raises four kinds of event and
knows nothing else about it; with no app wired up, crediting is a no-op.

## The pieces

| Piece | Where | What it does |
|---|---|---|
| Catalog | `vtsearch/achievements_catalog.py` | Pure data: `ACHIEVEMENTS` (each entry's `id`, `name`, `description`, `icon` and four ascending `tiers`), `TIER_NAMES`, the importers that do not count (`EXCLUDED_DATASET_IMPORTERS`), the streak's gap (`STREAK_GAP_SECONDS`), the media types and hours the two tick-off rows show, and the Readme Reader docs. Imports nothing from the app. |
| State machine | `vtsearch/achievements.py` | The `record_*` functions that credit events, `get_full_state()` (the panel's payload), the two watermark writers `acknowledge()` and `mark_toasted()`, and `wipe_state()`. Re-exports the catalog's public names. |
| Library seam | `vtscore/achievements_hooks.py` | `record_achievement(event, …)` for the four `KNOWN_EVENTS` (`vote`, `dataset_load`, `detector_import`, `find`), dispatched to whatever recorder the app registered. An unknown event raises at registration. |
| App wiring | `vtsearch/shim/__init__.py` | `register_app_achievement_recorders()` installs one recorder per event at startup, each forwarding to the matching `record_*`. |
| Routes | `vtsearch/routes/achievements.py` | `GET /api/achievements`, the two watermark POSTs, the phrase check and the raw-doc route; see the [API reference](API.md). |
| Frontend | `frontend/src/app/` | `AchievementsService` holds the state and the unlock stream; `achievementsRefreshInterceptor` refreshes it after any POST that can credit; `AchievementUnlockHostComponent` pops the toast; `AchievementsTabComponent` is the panel (Total Score, rows, the three expandable rows); `AchievementBadgeComponent` draws a badge at a tier; the trophy button and its dot are in `AppComponent`. |

## Where each count comes from

| Entry | Credited by | Raised from |
|---|---|---|
| `votes_cast`, `days_active`, `hours_voted`, `media_types_touched`, `detectors_trained`, `vote_streak` | `record_vote` (event `vote`) | `vtscore/state/votes.py`, on every vote that lands an item in a labeled state |
| `datasets_loaded` | `record_dataset_load` (event `dataset_load`) | `vtscore/datasets/load_origin.py`, when a load finishes, named by its importer |
| `detectors_imported` | `record_detector_import` (event `detector_import`) | `vtscore/labels/sync.py` when a pull from a labelset source changes a detector, and `vtsearch/routes/labels/importers.py` |
| `find_media` | `record_find` (event `find`) | `vtsearch/routes/detectors/scoring.py` (`POST /api/find-label`, the scoring pass behind Find and a sort by detector), `vtsearch/autofind.py` (AutoFind), and the CLI's autodetect in `vtscore/cli/scoring.py` |
| `docs_read` | `record_doc_phrase` | `POST /api/achievements/check-phrase` |

**One vote, six credits.** `record_vote` counts the vote, adds the detector to
`trained_detector_ids` the first time it is voted on, the media type to
`media_types_seen`, and the local day and hour to `days_seen` and `hours_seen`,
and moves the Marathoner streak. Each list is how a "distinct" count stays
distinct: an entry counts once, when it first joins its list.

**What is not a vote.** The vote path credits only a transition into a labeled
state: clearing a vote, or re-casting the vote an item already has, credits
nothing. Find's own labels on its results (`apply_labels_bulk_with_click_time`
with `record_achievement=False`) and the labels seeded from a detector's
example media are the system's, not the user's, and credit nothing either.

**Bulk votes do not build a streak.** Verified Good / Bad over a selection,
fill-from-sort and label import pass `count_streak=False`: they credit every
other vote-driven entry, but leave the running streak and its clock
(`last_vote_ts`) alone, so a hand-clicked run on either side of a bulk action
survives it and a thousand-item import is not a thousand-vote streak.

**Days and hours are the user's.** The frontend's `timezoneInterceptor` sends
the browser's `getTimezoneOffset()` as `X-Timezone-Offset`, and
`record_vote` buckets by that local clock, clamped to UTC±14. Without a
request (the CLI, a background thread) it falls back to UTC.

## State

Everything lives in the current user's per-user settings file, under the
free-form `achievement_state` key, written through `mutate_user` under the
settings lock. So it is per user, survives restarts, and travels with that
user's settings sync source like any other per-user setting.

```json
{
  "counters":  {"votes_cast": 137, "...": 0},
  "announced": {"votes_cast": 0,   "...": -1},
  "toasted":   {"votes_cast": 0,   "...": -1},
  "trained_detector_ids": [], "imported_detector_ids": [],
  "days_seen": [], "media_types_seen": [], "hours_seen": [], "docs_read_ids": [],
  "last_vote_ts": 0.0, "current_streak": 0
}
```

Nothing stores a tier: `get_full_state()` derives each entry's `tier_idx`
(`-1` for locked) from its counter and the catalog, so moving a threshold
re-tiers every user on their next read. `_ensure_state()` fills in any missing
key, which is why a new entry needs no migration: it starts at zero with both
watermarks at `-1`.

## The toast and the dot

Two independent watermarks per entry say how far the user has been told:

- **`toasted`** drives `pending_toasts`, every tier reached above it. The
  client pops a toast for each and calls `POST …/mark-toasted`, so a toast
  fires once per real unlock and never replays after a reload or restart.
  `AchievementsService` also remembers what it has emitted this session, to
  cover the round trip before the mark lands.
- **`announced`** drives `pending_announcements`, which lights the trophy's
  dot. It clears only when the user opens the panel, which acknowledges every
  pending tier (`POST …/acknowledge`). An acknowledged tier is also marked
  toasted, since a tier the user has read about needs no toast.

So the toast is fire-once whether or not the panel is ever opened, and the dot
stays lit until it is.

## Turning it off

`enable_achievements` is a per-user setting, on by default. While it is off,
every `record_*` returns without writing, `get_full_state()` returns the
payload of an empty state, and the frontend hides the trophy button, the
toasts and the panel. The settings route calls `wipe_state()` when the flag
flips from on to off, so counters reset on opt-out and a user who opts back in
starts from zero.

## Readme Reader

Four docs end in a `*Readme Reader code phrase:*` line: `README.md`,
`docs/user/USER_GUIDE.md`, `docs/CLI.md` and `docs/API.md`, each with a header
comment saying not to reword it. The phrases live only in the catalog's private
`_DOCS_RAW`. The server compares a SHA-256 of the submitted text, lower-cased
and with its whitespace collapsed, against the stored hashes, and never sends a
phrase to the client: the payload carries each doc's id, name, path and whether
it has been read. `GET /api/achievements/docs/<doc_id>/raw` serves the doc
itself, so the panel can link to it without reaching GitHub.
`tests/core/test_achievements.py` fails if a doc stops carrying its phrase.
Do not write a phrase anywhere else, this page included.

## Adding an entry

1. **Declare it** in `ACHIEVEMENTS`: a stable `id` (it keys the stored
   counters, so renaming one resets it for everyone), a `name`, the
   `description` the panel shows, an `icon` from the frontend's icon set, and
   four ascending `tiers`.
2. **Credit it.** Vote-driven counts go in `_credit_vote`; anything else gets a
   `record_*` function that checks `_is_disabled()` and writes through
   `mutate_user`. If the event happens inside `vtscore`, raise it with
   `record_achievement(event, …)`, add the event to `KNOWN_EVENTS`, and install
   its recorder in `register_app_achievement_recorders()`. If the frontend
   action that credits it is a POST the refresh interceptor does not watch, add
   its path to `WATCHED_PATTERNS`, or the toast waits for the next refresh.
3. **Test it** in `tests/core/test_achievements.py`.
4. **Regenerate what is drawn from the catalog:** the User Guide's table
   (`python scripts/gates/gen-docs-inventories.py`, which `./run-tests.sh` checks),
   and the slide figures (`python slides/figs/src/make-achievement-figs.py`,
   which refuses to run until its `COUNTS` has a line for the new id). Then
   re-read the slides' notes, which say how many entries there are.
5. **Queue a reshoot** of the `achievements` screenshot (see
   [`docs/reshoot-queue/README.md`](reshoot-queue/README.md)): the panel has a
   new row.
