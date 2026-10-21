# VTSearch Frontend Architecture

This document maps the Angular single-page app in `frontend/`. It is the
front-end counterpart to [ARCHITECTURE.md](ARCHITECTURE.md), which covers the
Python tiers.

It is a **map, not an inventory.** The SPA holds roughly a hundred components and
seventy-five services; enumerating them here would be a list that rots on the next
commit. Instead this document explains the handful of mechanisms everything
else is built on — the change-detection model, the service layer, the
active-dataset/detector context, the generated API client, and the component
composition conventions — so that any given file can be placed by reading its
neighbours.

**What lives elsewhere:**

| For | See |
|-----|-----|
| Build commands, dev server, Angular upgrade mechanics | [`frontend/README.md`](../frontend/README.md) |
| SCSS tokens, shared classes, copy style | [style-guide.md](style-guide.md) |
| REST endpoint reference | [API.md](API.md) and [api/](api/) |
| Backend state model, per-dataset contexts, auth | [ARCHITECTURE.md](ARCHITECTURE.md) |
| Modal back-button rules, desktop-only scope, budget policy | [`CLAUDE.md`](../CLAUDE.md) |

## Table of contents

1. [Shape of the app](#1-shape-of-the-app)
2. [Directory layout](#2-directory-layout)
3. [Feature areas and their boundaries](#3-feature-areas-and-their-boundaries)
4. [The service layer](#4-the-service-layer)
5. [Reactivity: the zoneless change-detection model](#5-reactivity-the-zoneless-change-detection-model)
6. [The active dataset/detector context](#6-the-active-datasetdetector-context)
7. [Talking to the backend](#7-talking-to-the-backend)
8. [Component composition conventions](#8-component-composition-conventions)
9. [Styling](#9-styling)
10. [Testing](#10-testing)
11. [Build and serve](#11-build-and-serve)
12. [Invariants](#12-invariants)

---

## 1. Shape of the app

Angular 21, TypeScript 5.9, **standalone components** (no `NgModule`s),
**zoneless change detection** (no `zone.js` anywhere — app, tests, or
`package.json`), built by the esbuild-based `@angular/build:application`
builder. Desktop-only: there are no responsive breakpoints and none should be
added.

Bootstrap is three files deep:

```
src/main.ts            bootstrapApplication(AppComponent, appConfig)
src/app/app.config.ts  the entire provider graph (below)
src/app/app.component.ts  persistent chrome + <router-outlet>
```

`app.config.ts` is the whole composition root:

```ts
providers: [
  provideZonelessChangeDetection(),
  provideRouter(routes),
  provideHttpClient(withInterceptors([
    timezoneInterceptor,        // X-Timezone-Offset
    activeContextInterceptor,   // X-Dataset-Id / X-Detector-Id
    achievementsRefreshInterceptor,
    errorInterceptor,           // toasts + offline circuit breaker
  ])),
]
```

Every service is `@Injectable({ providedIn: 'root' })` unless it is
deliberately component-scoped (see
[§8](#8-component-composition-conventions)), so there is no other provider
list to keep in sync.

`AppComponent` owns everything that outlives a route: the header (burger menu
of recent sessions, dataset/detector pulldowns, achievements/help/settings
buttons), the toast container, the offline banner, the dialog host, the
achievement-unlock host, and the app-level modals (Settings, Achievements,
Keyboard help, Add Dataset, New Detector, Find Results). Route content renders into
`<router-outlet>` inside `.main-content`.

**Routes** (`app.routes.ts`) are all lazy (`loadComponent`) and encode the
active context in the URL:

| Path | Guard | View |
|------|-------|------|
| `/dashboard` | — | Dataset & detector management |
| `/label/:datasetId/:detectorId` | `activeContextGuard` | Labeling / training |
| `/tune/:datasetId/:detectorId` | `activeContextGuard` | Tune (`FindViewComponent`) |
| `/browse/:datasetId` | `browseContextGuard` | VTSBrowse projection canvas |

Bare `/label`, `/test`, `/browse` (and anything unmatched) redirect to
`/dashboard`: a half-specified pair is not a representable state, so there is
nothing to render.

---

## 2. Directory layout

```
frontend/
├── angular.json            Builder wiring, production budgets, the dedicated `test` build config
├── ng-openapi-gen.json     Generated-client config (output: src/app/generated/api-client)
├── openapi.json            Committed snapshot of the Flask spec (see §7)
├── proxy.conf.json         Dev server: /api and /static → localhost:5000
├── vitest.config.ts        isolate: false — the cascade guard is in test-setup.ts (see §10)
├── docs-assets/            Symlink to docs/user/ — the in-app user guide is served from here
├── public/                 Favicons, logo, Toasty's faces (copied verbatim into the build output)
├── scripts/
│   ├── openapi-gen-cached.mjs   Hash-stamped wrapper around ng-openapi-gen
│   └── build-stamp.mjs          Writes generated/build-stamp.ts (the bundle's version, §7)
└── src/
    ├── main.ts, index.html, test-setup.ts
    ├── styles.scss         Global stylesheet entry; @use's the scss/ partials
    ├── scss/               _variables (design tokens), _layout, _components,
    │                       _data-table, _picker-shared — global, class-only rules
    └── app/
        ├── app.component.*, app.config.ts, app.routes.ts
        ├── components/     One directory per component; feature areas nest (see §3)
        ├── services/       API services, state services, coordination services (§4)
        ├── guards/         Route guards that resolve the URL pair into context (§6)
        ├── interceptors/   The four HTTP interceptors (§7)
        ├── directives/     Cross-cutting DOM behaviour (`no-focus-steal`, `panel-resize`)
        ├── models/         Hand-written types (§7): api.models.ts, projection.models.ts
        ├── utils/          Pure functions — no Angular DI, trivially unit-testable
        ├── testing/        Shared TestBed fragments and zoneless helpers (§10)
        └── generated/      **gitignored**; api-client/ (from openapi.json) and
                            build-stamp.ts, both rewritten on prebuild/pretest
```

Two rules keep this navigable:

- **A component's directory holds its children.** `dashboard/` contains
  `dataset-card/`, `new-detector-modal/`, `combine-datasets-modal/`, …
  because nothing outside the Dashboard uses them. A component used by two
  feature areas moves up to `components/`.
- **`utils/` is Angular-free.** Anything in there is a pure function over
  plain data (clip windows, progress formatting, grid-icon sizing, keyboard
  shortcut matching). If it needs `inject()`, it is a service, not a util.

---

## 3. Feature areas and their boundaries

Four routed views, plus the shared chrome. Their boundaries are worth knowing
because they determine which service a new piece of state belongs to.

### Dashboard (`components/dashboard/`)

The largest area by code volume. Dataset and detector cards, the sortable
managed-column tables behind them, per-row loading progress, and the entry
points into every creation flow (Add Dataset, New Detector, Combine…). It is
deliberately a **thin layout/wiring shell**: its state was lifted into
singleton services so the top-bar pulldowns can mirror it without depending on
the component —

- `DashboardColumnsService` — the two `ManagedColumns` instances (sort/visibility).
- `DashboardModalsService` — which row-action modal is open, for what target.
- `DashboardLoadingTasksService` — per-task loading rows, and the
  poll-until-settled-then-refresh bookkeeping.
- `DashboardSelectionService` — the highlighted table rows (which drive
  Train / Tune / Find / Combine / Delete), *owned* here rather than mirrored from the
  component, so the top-bar pulldowns show what is *selected* while the
  Dashboard is on screen (not merely what is loaded) by reading signals
  directly, and the selection survives a round trip to another view. Both
  grids share one `toggle(kind, id, additive)` ladder, and the ids the top bar
  reads are `computed` over the selection and the registry — the pulldown
  needs no push, and there is no per-mutation mirror call to forget.

The Add-Dataset and New-Detector flows are *not* owned by the Dashboard at
all: `NewThingFlowsService` is a singleton opener, and the modals are rendered
by `AppComponent` inside `@defer` blocks, so any surface can start those flows.

### Label view (`components/label-view/`)

The training loop. A three-panel layout (`vt-left-panel`, `vt-center-panel`,
`vt-right-panel`) with draggable dividers (the shared
`directives/panel-resize.directive.ts`), plus
`LabelViewPanelStateService` — a **component-provided** (not root) service
holding per-media-type panel preferences.

Two more component-provided services carry the view's non-chrome work:
`PairScopeService` (the pair lifetime and its reset) and `SortRunnerService`
(the sorts, and the `autoSelectNext` each one ends on). The component keeps
one-line forwarders for the handlers its template binds, so what is left in
`label-view.component.ts` is panel chrome, Autopilot's phase wiring and the
re-sort prompt. Neither service is `providedIn: 'root'`: the requests in them
are cancelled by the *component's* pair scope, which a singleton has no way to
name — see the note on `SortStateService` in `PairScopeService.seedBeta`.

Two ordering rules keep what the view serves independent of which request
answers first (#4318). Starting a sort ends whichever one is still in flight,
so the ranking on screen is the answer to the sort asked for last. And on entry
or after a pair switch the centre item is the view's own pick, made again
against every ranking and Select mode that lands until the user clicks, votes
or steps; from then on it is theirs, and no re-rank moves it. The backstop that
ranks a pair nothing else ranked (`seedRankingIfUnranked`) follows the first
rule: it reads the kind of the sort started last
(`SortRunnerService.newestSortKind`), not `sortMode`, and gives way only to a
learned one, so a text seed still in the air cannot hold it off (#4326).

**The typed query's hold** (#4736). A learned ranking is not always the one on
screen. While an Autopilot run that opened on a typed query has fewer labels
than the hold, the server marks its learned-sort payload `query_hold`, and
`SortRunnerService` keeps that ranking behind the text sort (`heldRanking`)
instead of installing it. Every pick reads `askWindow()`, the held ranking while
there is one: the Hard pick, New's atlas steer and the weak check's balance
(`askBalance`). So Autopilot asks what it would with the learned ranking on
screen, while the user sees the text sort's order, line and count. The first
payload without the mark takes the screen for good (`queryHandedOver`), and
stopping Autopilot ends the hold under the Sort `onAutopilotStop` leaves
selected (`endQueryHold`). Whether a run holds is the view's call
(`holdsTypedQuery`, label-view's `autopilotHoldsTypedQuery`): a run that opened
on the typed query's text sort, not an example opening, retrain mode or a
document dataset's walk.

The three panels are shared with the Tune view:

- **Left** — media list (virtual scroller), sort bar, Threshold, stripe
  overview, select mode, and the **Autopilot panel** that drives the automated
  vote → train → re-sort loop.
- **Center** — the media viewer, one child per media type (image, text, video,
  audio, document) plus the voting overlay.
- **Right** — labels, labelsets, vote grid, and the detector context bar.

**Folding the side panels** (#4673). Each side folds to a 48px strip
(`vt-side-panel-toggle`), per the `hide_left_panel` / `hide_right_panel`
settings, both on by default and shared by Train and Tune.
`PanelHideStateService` (root) reads them and holds a click until the PUT
echoes it. The left folds on an Autopilot tab only (`LeftPanelComponent.folded`;
Manual and Review are driven from the list), and in Train its strip is the
Autopilot panel's own phase rail. The right folds on every tab, and Tune opens
it on its own while a verdict is up (`FindViewComponent.rightCollapsed`). A fold
swaps the grid column, not the width: `--left-width` / `--right-width` keep the
open width, and the `.layout--*-collapsed` classes put the strip's 48px in its
place. A folded right panel is unmounted, which is why the views, not
`vt-right-panel`, own the votes poll.

**The Threshold** (`vt-balance`, the balance, in Train's Manual tab and on
both of Tune's tabs; #4413) is the one knob on the detector's line. The balance is
F-beta's beta: which way to lean between false positives and false negatives.
It is the detector's own, kept on the detector (#4665): the New Detector modal
mounts the same control (with no line, so no state line) and sends the pick
as `beta` on the create POST, starting on the user's `beta` setting, their last
pick. That is the only way Train's Autopilot tab, which has no Threshold of its
own, gets a balance the user chose.
The line is the set with the best estimated F-beta, and a spot check just says
what it estimated. Two values back the control, and they travel separately:
`SortStateService.beta` is the balance the radios show, seeded per pair by
`PairScopeService.seedBeta` (`GET /api/balance`); `balance` is the state of
the line on screen (`utils/line-balance.ts`, `LineBalance`: `unchecked`,
`checked`, or `gate` for a structural detector's verification gate, the count
kept, the check's two likely ranges and its F-beta estimate), and only ever
arrives *with* that line (a sort, a Tune pass, or the balance POST's own
response). Each view has one write path, both
`switchMap`-ed and pair-scoped so a balance the user moved past can never land
last. Tune's
(`betaRequests$`, debounced) installs the returned line straight over the
frozen scores. Label's (in `SortRunnerService`) re-runs the learned sort, but
only from the POST's response: the learned sort reads the balance server-side
and caches by it, so a re-sort that beat the POST would come back at the old
balance. When the balance keeps the same count of items before and after, the
line stays put and only its state is swapped: the count, not the state,
decides where the line sits. A text or example ranking draws its own line at
the balance too (#4603, #4732) but keeps its ranking, so Label does not re-run
it: from the POST's response it sends the ranking's `sort_token` to
`GET /api/sort/line` (#4760), `switchMap`-ed the same way, and moves only the
line and its count; the acquisition cut and the selection stay. A sort still
in flight when the balance lands is redrawn once it lands, since the server
may have drawn it at the old balance. The control is a False Positives - False
Negatives spectrum with three radios under its thirds (`BALANCE_PRESETS` in
`utils/line-balance.ts`: beta 4 toward false positives, 1 balanced, 1/4 toward
false negatives; #4448), and never shows a balance as a word or a number: each
radio's tooltip says where it sits (#4298, #4317). The radios show the host's
balance, never the click: a pick puts the DOM back on `value()` before
emitting, so a pick the host drops (Tune, mid-pass) leaves them where they
were. A stored balance off the list is shown on the nearest radio (nearest in
log space, a tie to the preset farther from 1, so the old 2 and 0.5 show as 4
and 1/4) and snapped to it through the control's
own `valueChange`, once `busy` (the host's `sortBusy`) is false, because Tune
drops a balance change while a pass is running. Under the spectrum the state
line reads "Checked the 40 kept you hadn't voted on · likely 55–100% right,
about half of them found (checked 5) · 32 more kept" or "32 more kept,
unchecked" (`balanceSummary`; the recall range is in words, `foundWords`, and
as a number only in the tooltip, `balanceExplanation`), with a green dot when
checked and a yellow one when not; there is no red. Nothing in it counts the
user's own votes, and it says so (#4914): the count is of the unvoted items
the line keeps, and the ranges cover the `keptAtCheck` items the line kept,
unvoted, when the check started (#4483; from a server before it, the top
`audited`): the walk samples the unvoted ranking, and recall's denominator is
the mixture's count of the positives left in it. The two counts differ once
the check's own votes are cast and the line re-drawn.
Under `advisory` that set is not the one kept, so the line names it rather
than setting its ranges beside the count as if they described the kept set.

**The spot check** (`SpotCheckRunnerService`, #4273, walking to the balance's
peak since #4413) measures that line, in Train only, and only Autopilot runs
it: it is Autopilot's **Spot Check** step, phase `spot` in
`AutopilotStateService`. The state's `spotCheck` says why one is running:
`weak` when the labels separate weakly, `done` for the check before Done. The
phase machine reaches `spot` the first time Done is due: every quality light
green, the labels line's level met (#4878), or the votes at the 200-vote cap
(#4903) (`doneChecked` latches it for the run once that check closes, so Done
follows), and a running check of either kind holds the phase at `spot`. A
run whose first labeling-status reading is already all green or at the level
(the Train view re-entered on a finished detector), or whose detector already
holds 200 votes, latches `doneChecked` at that reading, so
re-entry runs no check, even after a server restart has forgotten the last
one. The
Threshold control starts none (it has no check button), and Tune runs none:
it tests the balance Train set, and labelling more to set one is too late
there (#4317); its own test of the line is the Tune autopilot (below), and
the control's state line there is the host's (`lineState`: this corpus's
result, or untested) with the radios `locked` while a phase runs.
**When a check runs.** The label view starts both kinds. For weak separation
(#4496), an effect on the balance's `checkDue` (the server's `check_due`,
`weak_check_due` on the labels line's `separation`, only while no check has
ended, so once a session, #4909; never on a face detector, #4799) and
`sortBusy` calls `runDueCheck()`, which calls
`beginSpotCheck('weak')` once a learned sort has landed with `checkDue`
(under the typed query's hold, the held ranking's,
`SortRunnerService.askBalance`, #4736), never in Autopilot's opening
(`good`/`bad`/`more`, where no detector is trained) nor over another step.
For the check before Done, a second effect waits for the phase to reach
`spot` with no check running and the sort settled, then reads the balance's
`doneCheckDue` (the server's `done_check_due`: no check ended in the last 25
votes) and calls `beginSpotCheck('done')`, or `skipDoneCheck()` when a check
has just measured the line. Since the balance on hand right after a check
predates it, that effect also waits for the line re-read the check's close
starts (`awaitingLine`). The Threshold keeps a `.balance-due` note while a
weak check is due. Leaving `spot` sets the Select the step underneath uses and
does not re-sort; the completion hand-off follows the state stream
(`AutopilotPanelComponent`), since the check before Done ends outside any
vote. The step list shows Spot Check between Explore Diversity and Done; a
weak check makes it the active step and the list goes back to the step
underneath when it ends, as it does from Diversity to Boundary.
**In the vote flow** (#5136). The check is just another way to feed the
user the next item, so nothing on screen changes for it: no heading, no
button to stop it, no result. While `spotCheck` is set the label view's
advance (`advance()`) asks the component-provided `SpotCheckRunnerService`
instead of `SortRunnerService.autoSelectNext`, and the centre panel shows its
picks with the usual keys, undo and ↓ walk. The runner owns the check's
lifecycle against `/api/precision-check`. It draws the first round as the
step begins, but the item on screen stays: the next advance shows the first
pick, as a phase change waits for the next advance. Each vote on a pick is an
ordinary vote; once every pick of the round carries a label the round goes to
the check whole, with the labels the picks carry then, so a slip put right
with ↓ counts as corrected and an undone pick is shown again. That round trip
is the advance's (`advancePending`, so the voted item stays swiped off), and
no vote in the step re-sorts. Train's Autopilot tab shows no ranked list, so
no pick's rank is on screen. Stopping Autopilot (the effect on `spotCheck`)
or the view going away cancels a running check, first handing it the picks
already voted this round so they carry its provenance; a pair switch drops it
(`reset`), and the next advance draws a fresh one. The walk goes deeper while
its F-beta estimate does not fall and turns back once it does, and the check
is finished when `check.status` is `checked`. Finished, or refused by the
server, it ends the step on its own (`ended`): the view calls
`endSpotCheck()`, catches up on the votes, re-reads the line (a finished
check may have re-cut it server-side) with `GET /api/balance` through
`SortRunnerService.refreshLine`, and makes the advance the round owed under
the step underneath. Toasty has no hint for the step: there is nothing to do
differently. No re-sort follows the check's end: a retrain moves the list
under the result, which is what reports it `stale`, and the owner's model is
that *later* votes do that. The result lives in the balance control, on the
Manual tab. The line draws the same in every state; the state
and its likely range live in the balance control, and a stale range differs
only in its tooltip (`staleNote`).

### Tune view (`components/find-view/`)

The Dashboard's **Tune** button opens it at `/tune/:datasetId/:detectorId`.
It was called Find until #4525 and Test until #5098, and its code keeps the
first name, as the Train view's is `label`: `FindViewComponent`, `panelMode: 'find'`,
`VoteStateService.findMode`, and on the API side `/api/find-label`,
`find_mode` and the `find` SSE channel. **Find** is now the Dashboard's third
button, which opens no view: it starts a background AutoFind run per ticked
dataset with the ticked detectors (`DashboardComponent.onAutofind`,
`AutoFindService.find`), and the Find Results dialog opens as they land. The
dialog's **Browse** lays the listed items out in Browse through the same
`BrowseSubsetPrepService` the Tune view's Browse uses, with an origin
(`?from=results&run=…`) that sends Browse's Back to those results instead of
to Tune.

From the Tune click until the route activates, a bar stands under the button
in its reason's slot (`DashboardComponent.tuneBar`, #5098). It reads the
dataset and detector loads the `activeContextGuard` holds the route for off the
two loading-task channels (`utils/pair-load-bar.ts`), as one bar across both,
and is indeterminate while nothing has published yet, which also covers the
lazy route's chunk download.

Scores the active pair's dataset with its detector (`POST /api/find-label`)
and opens on two tabs in the left panel (`findTab`, owned by the view since
the centre and right panes switch with it; #4524): **Autopilot**, the Tune
autopilot, and **Review**, the ranked result in full (the work queue under
the line, the boundary walk, the Verified Good / Verified Bad piles, To
Dataset / Export / Browse, which can hand a subset of result ids to Browse
through `BrowseSubsetService`). Train's Autopilot / Manual split applied to
tuning: the guided flow is the default, the open one a tab away.

**The Tune autopilot** measures the line on the corpus in front of the user
with uniform picks from rank bands on both sides of it (`vtscore/docs/packages/training.md`;
the statistics are `vtscore/training/thresholds/line_test.py`, the routes
`/api/line-test*` in `vtsearch/routes/line_test.py`). The server owns the
sample and derives the phase on every read (`line_phase`), so nothing in the frontend accumulates it: the view's
component-provided `LineTestSessionService` holds the last `/api/line-test`
response plus the one thing the wire does not carry, the round *as dealt*
(the stage's dots show every pick of the round with its vote, where the
wire's `picks` are only the ones still pending). A vote goes out as it is
cast, a fast voter's next one while the last is in flight, and the ↓ key
sends an unvote; replies carry the whole state and can land out of order, so
each is stamped with its request's sequence and an older one never
overwrites a newer. The view starts the test once the pass lands and the
Autopilot tab is up (`startTestIfDue`), reloads it on the way back from the
Browser, and clears it with the pair state.

- `vt-line-test-panel` (left, projected into the left panel as
  `[findAutopilot]`, under the balance): Score, Check the matches, Check the
  misses, Done, in the shape of `vt-autopilot-panel`, each with one light
  (`widthLight`: red beyond twice the range's target width, yellow within
  twice, green at or under). The ranked list is not shown while a test runs:
  a pick is drawn from a band, and the list would show its rank (#4267).
- `vt-line-test-stage` (centre): the current pick with Good and Bad, the
  round's dots, the phase's prompt, through a `KeyboardService.captureVoteKeys`
  claim held while it is on screen.
  At Done it is the verdict (#5100), with the stale / moved note over it and
  its three exits (**Move to AutoFind**, which sets the detector's autofind
  flag and heads to the Dashboard; **Lean the Threshold**, the per-preset
  table the server prices from the same picks, whose click is the control's
  own balance pick; **Add Corrections and retrain**, the view's existing
  corrections path), all emitted to the view: the decision stays where the
  user just voted, and in view when the right pane is folded.
- `vt-line-test-result` (right): the result as it forms, in place of the two
  piles: the F-beta headline, the Right and Found ranges with their lights,
  the band-resolution precision curve and the picks by band. The trust chips
  (Training-domain overlap, Evidence coverage) and a 2×2 of the session's
  checks live under the result, read once the test is done.
- The balance is live between phases and frozen within one
  (`LineTestSessionService.locked`): the view drops a pick while a phase
  runs, and after a pick lands it re-reads the test so the result reports
  the line `moved`. **Add Corrections** marks it `stale` (the detector has
  now seen the test set; the server's `find_eval_stale`).
- The verdict is kept on the detector server-side (#4526), so a `start` over
  an unchanged ranking can come back already `done` with `test.kept_at` set:
  the result pane says when those picks were taken, and the kept picks are
  session votes again, in Review's piles once it opens. There is no
  forget button: a retrain marks the verdict stale and a changed ranking
  deals a fresh test, so nothing is owed to the user there
  (`/api/line-test/forget` is the screenshot harness's reset). The two readers of the kept verdicts are the
  detector Stats modal's *Tested on* section and the AutoFind tab's
  `vt-detector-card`, which shows its row's `test_verdict` under the name;
  both word it through `verdictLine` / `verdictTitle` in `utils/line-test.ts`.

**What a vote waits on** (#5099). The view's one `MediaPrefetchService`
effect warms the next two images of whichever queue the tab votes through:
the round's unvoted picks after the one on screen
(`LineTestSessionService.upcoming`) on Autopilot, the boundary walk on
Review. At a round boundary that queue runs on into the next round, which
the server deals once one pick is left, both ways that pick can go
(`test.ahead`, #5116): `upcoming` takes the two ways' first picks in turns
while the pick is unvoted, and the way it went once it is voted, so the
next round's first image is held before the vote that deals it. The vote piles (`/api/votes`) are read when the view opens, when
Review opens, and after a scoring pass, a balance change or a pair switch,
and polled only while Review is up (`syncPiles`): no vote on either tab
re-reads them. The Find view reads them as `?scope=verified` (#5115):
the full read carries every item's presumed label, megabytes on a large
dataset, where the verified scope is a few KB. `VoteStateService` picks the
scope from `setFindMode`, so in Find `goodVotes` / `badVotes` hold verified
items only (plus votes still in flight), and the unverified items are placed
by the frozen ranking and the cutoff: `unverifiedSortOrder`, the boundary
walk, and the right panel's *Unverified Good* / *Unverified Bad* counts
(`FindViewComponent.unverifiedCounts`).

### Browse view (`components/browse-*`)

VTSBrowse: a UMAP projection rendered on a canvas as a hex-tile pyramid. This
area is unusual and mostly self-contained — `browse-canvas` is the single
largest component in the app because it owns the render loop. Its state is
split across small services rather than living in the canvas:
`BrowseViewportService` (visible region, shared with the minimap) and
`BrowseSelectionService` (selection at *item* granularity, so it stays coherent
as bins split and merge across zoom levels) — both provided on `browse-view`,
not root — plus the root `TileCacheService`,
`BrowsePrepService` (load + project before navigating), and
`ProjectionApiService`.

### Shared chrome

`AppComponent`'s header, `context-pulldown` (the dataset and detector
pickers, plus the incompatible-pair explainer), `toast-container`,
`dialog-host`, `offline-banner`, `achievement-*`. Anything here must work on
every route.

**Where does new code go?** If it is used by one view, nest it under that
view's directory. If two views need it, promote it to `components/` and give
its state a root service. If it is state two *services* need, it belongs in a
service of its own — the pattern the Dashboard and Browse areas already
follow.

---

## 4. The service layer

`services/` holds three distinct kinds of file. The suffix tells you which.

### `*-api.service.ts` — thin typed HTTP wrappers

One method per endpoint, returning an `Observable` of a generated type. They
hold **no state**. Each wraps a function from the generated client rather than
hand-building a URL, unless the spec cannot type the call; those stay on plain
`HttpClient` with a comment saying why (a plugin-field route whose body is the
plugin's, a multipart upload, a blob download, the Browse projection
endpoints):

```ts
getMediaIds(): Observable<MediaIdsListResponse[]> {
  return listMediaIds(this.http, this.config.rootUrl).pipe(map((r) => r.body));
}
```

They are split to roughly mirror the backend's route modules
(`vtsearch/routes/datasets/{listings,registry,ui}.py` →
`datasets-{listings,registry,ui}-api.service.ts`, and so on). The mirror is
approximate and does not need to be exact; the split exists to keep any one
file readable.

### `*-state.service.ts` — client-side state

The stores the views bind to: `MediaStateService` (the dataset-wide media stub
list and current selection), `VoteStateService` (the per-pile vote sets, undo
and redo), `SortStateService`, `SettingsStateService`, `DatasetStateService`
(the registry listing), `LabelsetStateService`, `AutopilotStateService`.

They own state and the reactive surface over it; they call API services for
I/O. A component should almost never call an API service *and* keep the result
in a field that another component also needs — that is a state service.

### Coordination services — everything else

The ones worth knowing before changing anything:

| Service | Responsibility |
|---------|----------------|
| `ActiveContextService` | The active (dataset, detector) pair, split into intent/active layers ([§6](#6-the-active-datasetdetector-context)) |
| `ContextSwitchService` | Drives a pair change end to end: intent → load → promote to active |
| `ActiveContextWatcherService` | Reacts when an active id disappears from the registry (deleted elsewhere) |
| `PairRecoveryService` | Loads the active pair again when the backend has lost it (a restart), so an open Train, Tune or Browse view comes back instead of 409ing; sends a lost detector-positives map back to the Dashboard ([§6](#6-the-active-datasetdetector-context)) |
| `ProgressEventsService` | The single `EventSource` over `/api/events`, fanned out per channel |
| `ConnectionStateService` | Online/offline, and the probe that reopens the circuit breaker |
| `ToastService` | Toasts: four levels, the structured `ErrorContext` from failed requests, and the backend's `notification` channel |
| `VtDialogService` | `confirm()` / `prompt()` as promises, rendered by `dialog-host` |
| `NewThingFlowsService` | Singleton openers for the Add-Dataset / New-Detector flows |
| `AutoFindService` | The end of a background AutoFind: opens the results of a Find this tab started (the Dashboard's big **Find** button, one run per ticked dataset) unless the dialog already shows another, toasts any other run of the user's (an import's, a dataset's ⋯ **Run AutoFind**, or a Find that landed while the dialog was open) with a **View Results** action, and holds the run the app-root Find Results dialog shows. AutoFind is the unattended path and never opens the dialog on its own (#4615) |
| `MediaMetadataCacheService` | Lazy batched fetch of full metadata for whatever is in the viewport |
| `PairScopeService` | **Component-provided** (`find-view`, `label-view`): the active pair's lifetime, the `scoped()` teardown operator, and the pair-change reset in its one correct order |
| `SortRunnerService` | **Component-provided** (`label-view`): runs the sorts — text, learned (with its job poll), detector, example — and advances the selection they end on. Lives beside the view rather than on the root-singleton `SortStateService` because every call in it is torn down by `pairScope.scoped()` |
| `KeyboardService`, `ThemeService`, `AchievementsService`, `AuthService` | App-wide concerns wired in `AppComponent` |

`adaptive-poll.ts` is not a service but a shared operator; see
[§7](#7-talking-to-the-backend).

---

## 5. Reactivity: the zoneless change-detection model

**This is the section to read before touching any state.**

The app runs `provideZonelessChangeDetection()` and `zone.js` is gone
end-to-end. There is no monkey-patched `setTimeout`, no zone tick after an
HTTP callback, no implicit "something happened, re-render everything". Angular
refreshes a view only when it is *notified*, and the notifications are:

1. A **signal** read during that view's template evaluation is written.
2. A **template-bound event** fires (`(click)`, `(widthChange)`, …).
3. `AsyncPipe` emits, or an `@Input`/signal input changes.
4. Something explicitly calls `markForCheck()` / `ApplicationRef.tick()`.

Anything else that mutates rendered state produces a **stale view** — the
value changed, the DOM did not. That is the characteristic frontend bug in
this codebase, and it is silent.

### The rules

- **State that a template reads must be a signal**, or must reach the template
  through `AsyncPipe` / `toSignal`. Writing a plain field from an HTTP
  continuation, a timer, an `EventSource` callback, or a promise `.then()`
  notifies nobody.
- **Getter-over-signal is the sanctioned migration shape.** Several state
  services expose `get sortBusy() { return this._sortBusy(); }` over a private
  signal, so existing `sortState.sortBusy` bindings stayed byte-identical while
  becoming reactive. A signal read through a getter during template evaluation
  *is* tracked as a dependency of that view —
  `testing/getter-signal-zoneless.spec.ts` pins this.
- **Per-media-type settings prefs go through `SettingsStateService.perMediaType`.**
  A `{media_type: value}` settings key bound to a media-type signal returns a
  `computed` value plus a merge-preserving setter; do not hand-roll a shadow
  `Record` field kept in sync by `effect()`s. The value is a real signal (a
  template binding on it repaints on its own), and there is no mirror to go
  stale when a key disappears server-side. The setter **merges**; a setter
  that replaced the dict would wipe every other media type's preference. It takes a plain key, or a *signal* of a key where
  the key itself varies (`vt-view-controls` picks `focus_mode_left` /
  `_right` / `_popup` from its `side` input). `LabelViewPanelStateService` is
  the reference use; `find-view`, `browse-view`, `browse-bin-popup`,
  `browse-selection-panel`, `right-panel` and `view-controls` are the other
  consumers.

  Two notes on where the seams are. **The Settings modal is deliberately not a
  consumer**: it edits a *draft* settings signal across *every* media type via
  a `typeId` loop, not "the current one", so the helper's shape does not fit.
  And a preference whose control derives its next value from the current one —
  the `vt-view-controls` size buttons, which also grey out at the ends of the
  ladder — keeps a **local optimistic signal** written on click, with the
  preference behind it; binding straight to the preference would make a rapid
  second click read a pre-round-trip value.

- **Consumers use `effect()`, not `subscribe()`.** A constructor `effect()`
  reading a signal auto-disposes with the component, which is why the
  teardown plumbing has been dropped wherever it was the last user.
- **Where a subscription is still needed, tear it down with
  `takeUntilDestroyed()`** — see "Subscription teardown vs. cancellation"
  below for the one distinction that decides it.
- **Reads go through `rxResource`.** The read path is being migrated onto
  Angular's reactive resource primitives (`SettingsStateService`,
  `MediaStateService`, several picker modals so far). `rxResource({ params,
  stream })` wraps the *existing* generated-client method, so the typed client
  and interceptor chain are untouched and the service's public surface becomes
  a value signal (e.g. `settingsSignal`, a `computed` over `resource.value()`)
  plus `isLoading()` / `error()`. Prefer it over raw
  `httpResource`, which would bypass the generated client. The remaining
  conversion recipe and the list of still-open call sites live in
  [`docs/plans/httpresource-migration.md`](plans/httpresource-migration.md).
- **Mutations, SSE and pollers stay imperative** — POST/PUT/DELETE remain
  plain `HttpClient` calls; write the server's response back into the store.
  This is a deliberate boundary, not an unfinished migration.
- **Every component is `ChangeDetectionStrategy.OnPush`.** All of them, with
  no exceptions today. A new component should be too.
- **Callbacks from outside Angular must write a signal.** The SSE pump, canvas
  RAF loops, and the drag handlers that deliberately run outside Angular
  (`directives/panel-resize.directive.ts`) all reach the UI either by writing a
  signal or by emitting through a template-bound output. `NgZone.run()` is a
  no-op here and is not the fix.
- **A `BehaviorSubject` service is not wrong, but its bridge is load-bearing.**
  Roughly a third of the services still hold RxJS subjects; each is
  change-detected via its `toSignal`/`AsyncPipe` bridge. A missed bridge is a
  stale-view bug, not merely a slow one.

`AsyncPipe`, `toSignal`, and `toObservable` are all in use and all fine. What
is never fine is a rendered value with no notification path.

---

## 6. The active dataset/detector context

The backend resolves per-dataset and per-detector state from the
`X-Dataset-Id` / `X-Detector-Id` request headers (see
[ARCHITECTURE.md § Multi-dataset support](ARCHITECTURE.md#multi-dataset-support)).
The frontend's job is to make sure those headers name a pair the backend has
actually loaded. Five pieces cooperate.

### `ActiveContextService` — two layers, not one

```
intent   what the user just picked (pulldown click, deep link)  → UI affordances
active   what the backend has loaded                            → HTTP headers
```

The split exists to fix a real cascade: tagging requests with the new ids the
instant a pulldown was clicked produced a storm of `409 dataset_not_loaded`
until the load finished. `intent` updates immediately so the pulldown
highlights; `active` lags until a load completes and is promoted explicitly.

- Read the pair as a whole via `pair$` / `intentPair$` (subscribing to the two
  halves separately fires twice for one atomic change).
- `setActivePair()` writes both layers — for cleanup paths only (registry
  watcher, `clear()`, tests).
- `nextRequestId()` / `currentRequestId` implement **latest-caller-wins**: a
  prep step captures the id at start and discards its result if the id has
  moved. Cancelling in-flight work is best-effort; this check is the
  correctness guarantee.

### `ActiveDatasetService` / `ActiveDetectorService` — the ids as *entries*

`ActiveContextService` carries **ids**, on an RxJS layer a `computed` can't
track. Almost every consumer wants the registry *entry* behind the id — its
name, its media type, its embedder — so each one used to redo the same
`datasets.find(d => d.id === …)` by hand. That read whatever happened to be
loaded at call time and never updated when the registry landed a moment later:
the lifecycle gap that left an export filename detector-less when the modal
opened first (#2819).

The two services close it, one per half, with the same five members:

```
activeId      the id the backend has loaded ('' when none)
intentId      the id the user picked
datasetId /   activeId || intentId — name the user's pick immediately
  detectorId  rather than blanking for the duration of a load
dataset /     the registry entry, or null (nothing selected, or the
  detector    registry fetch is still in flight)
datasetName / the entry's name, or ''
  detectorName
```

Reach for these from **components**: a `computed`/`effect`/template read
repopulates on its own once the registry or an in-flight switch settles.

Two places deliberately keep the imperative lookup, and should:

- **Route guards** run once, before activation, and have already awaited the
  registry — there is nothing left to arrive.
- **Pre-reactive services** (`ActiveContextWatcherService`) resolve the pair
  from the arrays their own `combineLatest` emitted; reading a signal instead
  would resolve against a *different* snapshot than the one being handled.

Both still index rather than scan: `DatasetStateService` exposes `datasetById`
/ `detectorById`, `computed` Maps over the two registries. They are the shared
lookup the services are built on, and reading one is signal-tracked exactly
like reading `datasets` — so swapping a `find` for a `.get` is a readability
change, never a reactivity one.

Note what these services are *not* for. A lookup that resolves something other
than the active pair — the Dashboard's multi-select, the id being switched
*to*, an entry from an HTTP response body — is a different question that
happens to share the `find(d => d.id === …)` shape. Use the Maps there if the
predicate is a plain id match; do not route it through the active-context
services.

### `ContextSwitchService` — the only way to *change* the pair

`switchTo()` tags intent, kicks off whatever dataset/detector loads are
needed, and promotes to active only once they settle. It has cancel-and-replace
semantics and a `switching$` observable for loading UI. A failed load must not
promote — that would re-open the 409 cascade.

### `activeContextInterceptor` — the header attachment

Reads the **active** layer only and sets each header when its id is non-empty.
That is the whole interceptor; all the difficulty lives upstream in the
intent/active split.

**Native element requests bypass it entirely.** `<img src>`, `<audio src>`, and
`<video src>` are browser requests, not `HttpClient` requests, so they carry no
headers. Build those URLs with `ActiveContextService.mediaUrl(path, extra?)`,
which appends `dataset_id` / `detector_id` query params from the active layer.

**Reading an item of another dataset.** Show in photo (#4750) shows a face
crop's photo out of the Image dataset the same import produced, in an overlay,
while the Face pair stays active: switching the pair would land on the
incompatible-pair explainer (a face detector cannot pair with an Image dataset)
and reset the face ranking. Such a request names the other dataset **and no
detector**: `DATASET_OVERRIDE` on an `HttpClient` request (the interceptor sets
`X-Dataset-Id` to it and drops `X-Detector-Id`), `mediaUrlIn(datasetId, path)`
for an `<img src>`. Never send the active detector with another dataset's id.
The backend rehydrates a detector's votes whenever a request pairs it with a
dataset other than the one they were derived for
(`ensure_votes_match_active_dataset`), so one such request clears the session
the user is in.

Collapse to photos (#4751) follows the same rule for a whole grid: the left
panel swaps `vt-media-list` for `vt-photo-grid` while `CollapsePhotosService`
is active (switched on, on a dataset where Show in photo is offered), and the
grid draws each photo with `mediaUrlIn(photoDatasetId, …)`. The grouping is the
server's (`GET /api/sort/collapsed`, keyed by `SortStateService.sortToken`, so
Tune's `setSortResults` keeps find-label's token); the grid only pages rows and
leaves out the faces it is told to hide. A face on a photo emits the same
`mediaSelect` / `mediaVote` / `mediaContextRequest` a card does, so the host's
vote flow is unchanged. The Show in photo overlay and the grid both answer the
409 for an unloaded Image dataset with `SiblingLoadService`
(component-provided), which loads it and calls back.

### Routes and guards — the URL is the source of truth

`/label/:datasetId/:detectorId` and `/tune/:datasetId/:detectorId` carry the
pair so reload, share links, and browser back/forward all work.
`activeContextGuard` resolves the URL pair before the view renders: it awaits
the registry fetch (a cold deep-link can arrive first), toasts and redirects to
`/dashboard` for an unknown id, fast-paths when the pair already matches, and
otherwise holds activation on `ContextSwitchService.applyActivePair(...)`. An
*incompatible* pair (mismatched media types) is allowed through — the
`vt-incompatible-pair-explainer` overlay is a legitimate UI state.

`browseContextGuard` is the dataset-only variant, with one carve-out:
ephemeral detector-positive contexts (`__detpos__<id>`) are built server-side,
are deliberately absent from the registry, and skip the checks. The guard
still makes the map the active dataset, so a reload names it as the
Dashboard's Browse action does.

Finally, `ActiveContextWatcherService` watches the registry against the active
pair and, when an id disappears (deleted from another tab, the CLI, or another
session), clears the affected half, toasts, and bounces off any now-broken
view.

`PairRecoveryService` covers the other way a pair goes missing: still
registered, but no longer loaded, because the backend restarted under an open
tab (#5155). Every request then answers 409 `dataset_not_loaded` /
`detector_not_loaded`, and nothing else would load the pair again: the guard
let the view in long ago, and polls skip a failed tick. `errorInterceptor`
hands it each failed response, including the ones that opt out of the toast,
as Browse's tile requests do; a 409 naming the active pair, with no switch in
flight and the tab on that pair's `/label` or `/tune` view or on `/browse` of
its dataset (#5158), starts `ContextSwitchService.reloadActivePair()`, and its
`reloaded$` makes the view run its pair-change reset. For Browse that reset
has to ask for a build: a dataset load leaves the pyramid out of memory, and
the build restores the layout persisted in the dataset's container rather than
fitting it again. The reload loads the dataset *before* the detector,
because the detector load resolves its dataset from the request's
`X-Dataset-Id`, which during a reload names the unloaded dataset. A 409 that
lands mid-switch or mid-reload, or for a request sent before the last reload
landed, is absorbed without a toast; one off the pair views still toasts. A
reload that fails, or a pair lost again within a minute of one, sends the tab
to the Dashboard with a message saying to reopen it.

A detector-positives map has nothing to load it again from: its context lived
only in the memory of the process that built it. After a restart the backend
answers 404 `dataset_not_found` for its id, not 409, so `PairRecoveryService`
answers that 404 apart (#5159): with the tab on `/browse/__detpos__<id>`, it
goes to the Dashboard with a message saying to browse the detector's positives
again from there. The 404 is absorbed off that view too, where it can only be
a request sent before the tab left. `dataset_not_found` for any other dataset
is untouched.

---

## 7. Talking to the backend

### The generated OpenAPI client

`frontend/openapi.json` is a **committed snapshot** of the Flask-smorest spec.
`ng-openapi-gen` turns it into `src/app/generated/api-client/` — typed models
plus one function per operation. The generated tree is **gitignored** and
rebuilt automatically:

```
prebuild / prebuild:prod / pretest / pretest:ci
    → node scripts/openapi-gen-cached.mjs && node scripts/build-stamp.mjs
```

The wrapper hashes the spec, the generator config, and the generator's own
version, and skips regeneration when nothing moved (`--force`, or
`npm run generate-api-client`, regenerates unconditionally).
`build-stamp.mjs` writes the bundle's version into
`generated/build-stamp.ts`; `BuildSkewService` compares it with
`GET /api/version` at startup and raises a toast when a new server is serving
a stale SPA (the mechanism is described under "Versioning" in `CLAUDE.md`).

The snapshot itself is gated: `./run-tests.sh` re-dumps the spec from the
running Flask app and **fails if `frontend/openapi.json` is stale**. When you
change a backend schema or route, run `npm run regenerate-openapi-snapshot`
and commit the result. That gate is what makes a backend field rename a
TypeScript compile error instead of a runtime surprise.

The client is configured with `apiService: false` — there is no generated
god-service. Hand-written `*-api.service.ts` files call the generated
functions, which is what lets them stay thin while keeping the typed contract.

### `models/api.models.ts` — what stays hand-written

Everything the spec describes is **re-exported** from the generated tree, not
mirrored. Only three categories are hand-written, because the spec cannot
describe them:

- **SSE payloads** (`ProgressEvent`, `LoadingTask`, …) — `/api/events` is a raw
  streaming response flask-smorest does not model.
- **Client-only shapes** that no endpoint returns.
- **Request-side settings shapes** the frontend builds and posts.

Adding a hand-written interface that duplicates a generated one is a
regression: it can drift, and the compile-time guarantee is exactly what is
lost.  `models/projection.models.ts` follows the same rule for the Browse
payloads: it re-exports the generated `ProjectionMeta`, `TileResponse`,
`RegionLabel`, … and hand-writes only `ProjectionLayout` (a ready meta, whose
layout fields the one shared schema leaves optional), `BinShape` and the
canvas's `ViewTransform`.

### The interceptor chain

Order matters; it is fixed in `app.config.ts`.

1. **`timezoneInterceptor`** — attaches `X-Timezone-Offset` so the backend can
   bucket achievement milestones by the user's wall clock, not UTC.
2. **`activeContextInterceptor`** — the context headers ([§6](#6-the-active-datasetdetector-context)).
3. **`achievementsRefreshInterceptor`** — after a POST to any endpoint that
   could unlock a tier (vote, find, label import, dataset load), schedules a
   coalesced achievements refresh.
4. **`errorInterceptor`** — the failure chokepoint. Turns every failed response
   into a structured `ErrorContext` toast (endpoint, status, `request_id`, and
   the active pair at the time), deduped by status + message so one backend
   condition surfacing from several in-flight requests shows once. It also
   implements the **offline circuit breaker**: while offline, requests
   short-circuit to a synthetic network error without touching the wire, which
   is what stops the console filling with `ERR_CONNECTION_REFUSED`.
   `ConnectionStateService`'s probe (tagged with the `CONNECTION_PROBE` context
   token) is the one request allowed through.

Opt a single request out of the global toast with the `SKIP_ERROR_TOAST`
`HttpContextToken` — for probes that expect a 404, retry loops, and forms that
render their own inline validation. The error still propagates to
`catchError`/`error:` handlers; only the toast is suppressed.

### Server push, and polling when push won't do

`ProgressEventsService` holds **one** `EventSource` on `/api/events` and fans
it out to every consumer.
Channels: `server` (a per-connect `boot_id`, so a backend restart fires
`serverReset$` and consumers can drop state keyed on dead `task_id`s);
`loading-tasks` and `detector-loading-tasks` (`LoadingTask[]` lists); and
`sort`, `find`, `eval`, which carry the `ProgressEvent` shape that
`utils/format-progress.ts` renders. A payload-less `heartbeat` keeps the
circuit breaker online during long silent operations. The channel state is held in
signals — a signal write inside the SSE callback notifies Angular's scheduler
directly, which is precisely what makes the pump work without zone.js.

One channel breaks that mould: **`notification`**, carrying one-off messages
any server-side code (most usefully a plugin that hit a recoverable problem
and carried on) pushes with `notify()`. Those are *events*, not state — there
is no "current notification" to re-read and none is replayed to a late
listener — so the service exposes them as `notifications$`, a `Subject`
rather than a signal, and `ToastService` subscribes to turn each into a toast
of the matching level. See [`api/events.md`](api/events.md) for the payload
and [`EXTENDING-plugins.md`](EXTENDING-plugins.md#notifying-the-user-toasts)
for the producing side.

### Subscription teardown vs. cancellation

Two different jobs get done with the same RxJS machinery, and conflating them
is how a working poller gets swept into a bug. Name which one you are doing
before reaching for an idiom.

**Teardown** — "stop this when the component dies." There is exactly one
sanctioned idiom, and new code must use it:

```ts
private destroyRef = inject(DestroyRef);
...
this.someService.thing$.pipe(takeUntilDestroyed(this.destroyRef)).subscribe(...);
```

Drop the explicit `destroyRef` argument only in an injection context (a field
initializer or the constructor body); anywhere else — `ngOnInit`,
`ngAfterViewInit`, an event handler — it must be passed. The point of the
idiom is that teardown is *declared at the subscription*, so adding a
subscription can never forget to add a matching `unsubscribe()`. It is now the
**only** teardown idiom in the SPA: the hand-rolled `destroy$ = new
Subject<void>()` fired from `ngOnDestroy` and the `subs: Subscription[]` array
drained in `ngOnDestroy` are both gone, and neither may be reintroduced. Both
were strictly worse — the bookkeeping was manual and the compiler did not check
it — and the subject form had a hazard the sanctioned idiom does not: RxJS
`takeUntil` never fires on a pre-completed notifier, so a subscription armed
*after* `ngOnDestroy` ran was never torn down at all. `takeUntilDestroyed`
completes such a subscription immediately.

Because teardown rides on the `DestroyRef`, a **spec drives it with
`fixture.destroy()`**, never by calling `ngOnDestroy()` by hand — a bare hook
call runs only what the hook itself still does.

**Cancellation** — "stop the *previous* one because a newer one supersedes
it." This is not teardown and `takeUntilDestroyed()` cannot express it; the
component is very much still alive. It legitimately takes one of two shapes,
and both are correct as written:

- A **scope subject** fired on each reset, for a family of streams that share
  a lifetime shorter than the component's: `polling$`
  (`DashboardLoadingTasksService`), `stopPolling$` (`VoteStateService`) and
  `sortSuperseded$` (`SortRunnerService`) cancel superseded work. Name it for the
  scope it bounds, never `destroy$`.

  The dataset/detector pair's scope is the one case where the subject is not a
  component field: `PairScopeService` (component-provided on `find-view` and
  `label-view`) keeps it private and exposes `pairScope.scoped()` as the pipe
  operator, because the *ordering* around firing it is load-bearing. A pair
  switch must supersede **before** any of the new pair's state is installed, or
  a late response repaints the old pair's ranking over the new one's — a silent
  wrong-results bug, not a crash. `resetForNewPair()` is the only caller that
  can fire the scope, so the order is not a caller's to get wrong. The service's
  own `ngOnDestroy` covers destroy-time teardown, so a view must not (and need
  not) fire the scope by hand.
- A **re-assigned `Subscription` field** that unsubscribes the previous value
  before storing the next: `text-viewer`'s `sub`, `folder-browser`'s
  `currentSub`, `browse-bin-member-grid`'s `scrollSub` (re-keyed to the
  viewport *instance*), `label-importer-modal`'s `ingestSub`.

A `Subscription` field is only a teardown idiom when it is written once and
read only by `ngOnDestroy`. If it is re-assigned anywhere, it is cancellation
— leave it alone.

Where a real poll is still needed, reach for one of the two primitives rather
than `timer(0, n)` + `switchMap`. Both run each request to completion before
scheduling the next — a `switchMap` timer cancels in-flight requests, so a
backend slower than the interval froze the panel permanently (#2572) — and
which one you want is decided by whether the thing being polled *ends*:

- **`adaptivePoll()`** (`services/adaptive-poll.ts`) for open-ended background
  polling. Eases from `fastMs` to `slowMs` after N unchanged responses, and
  suspends entirely while the tab is hidden. Callers own teardown via
  `takeUntil(stop$)`.
- **`pollUntil()`** (`services/poll-until.ts`) for a job the user is waiting on:
  a projection build, anything with a terminal state. The caller's `apply`
  returns `'continue'` or `'stop'`, and the loop tears itself down when it
  settles. A failed request is absorbed and retried with exponential backoff
  (2s → 30s); only five consecutive failures give up, via `onLostContact`.
  Teardown is the returned handle's `stop()`.

`adaptivePoll`'s cadence easing and pause-while-hidden are precisely wrong for
the second case — progress is never stale while a build runs, and pausing in a
background tab would strand the user behind a bar that stopped moving — which
is why they are two helpers rather than one with a flag.

---

## 8. Component composition conventions

- **Standalone, `OnPush`, `vt-` selector prefix.** (`AppComponent` is the sole
  `app-` selector, as the bootstrap root.)
- **Signal inputs and outputs** (`input()`, `output()`) for new components;
  older decorators survive in places and are converted opportunistically.
- **Presentational children, stateful parents.** The larger pickers are
  explicit about this: `vt-source-picker` takes precomputed lists as inputs,
  emits user actions as outputs, and lets the parent decide what they mean —
  which is how the same widget serves both the Add-Dataset modal (run an
  importer) and the New-Detector modal (materialise one example file). Layout
  that differs between hosts is projected into named content slots rather than
  branched on inside the child.

### Modals

`vt-modal` (`components/modal/`) is the shared shell: backdrop, CDK focus
trap, close button, and a module-level **stack** of open instances so Escape
dismisses only the topmost one. Modals genuinely nest here (New Detector →
media crop, Settings → importer picker, anything → a confirm dialog), and
without the stack one keypress closed them all and lost the outer form.

- `dialog-host` + `VtDialogService` provide promise-returning `confirm()` /
  `prompt()`, including the standardised `confirmDestructive()` phrasing.
  Purely informational messages go to `ToastService`, never a modal.
- **Back vs Cancel is a rule, not a preference.** An inner view that *replaces*
  an outer one gets a left-aligned `← Back` chevron; `Cancel` in the footer
  means "abandon the whole dialog". A picker whose tab bar stays visible has no
  outer view to return to and correctly has no back button. The full rule,
  including the canonical markup and the persistent-tab exception, is in
  [`CLAUDE.md`](../CLAUDE.md) under "Nested-modal back buttons".
- **Plugin-field forms preview their template variables.** A modal that builds
  a form from a plugin's `fields` (Export, Find Results) seeds each
  value through `PluginTemplateVarsService`, which resolves the *declared*
  `template_vars` — `{detector_name}`, `{username}`, the date parts — so the
  user sees and can edit the value the server would substitute rather than a
  raw placeholder (issue #3199). Only declared names are touched, and anything
  unresolvable stays templated for the server. Never do this in a form whose
  values are **persisted** (AutoFind's saved exporter fields, a labelset-sync
  source): those templates are meant to re-resolve on every later run. See
  `utils/plugin-template-vars.ts`.

### Lazy loading and the bundle budget

Routes are lazy by construction. Beyond that, the heavy app-level modals are
imported by `AppComponent` but used **only inside `@defer` blocks**, so Angular
splits them into separate chunks — they drag in the file browser, the crop
modal and `marked`, which together push the eager bundle over budget.

The production `initial` budget is deliberately tight and the rationale is
written into `angular.json` next to the number. `./run-tests.sh` treats **any**
`▲ [WARNING]` from `build:prod` as a hard failure, including the 8 kB
`anyComponentStyle` budget. Fix the bloat — split the component, extract shared
styles, delete dead rules. Raising a budget needs the user's explicit approval.

What actually sits in the eager bundle is mostly **framework**: 317 kB of
Angular against 121 kB of app code when it was last audited (#3811). That is
why the budget carries visible headroom — Angular's own patch releases move the
number a few kB at a time, and when the margin was under 1 kB a routine
security bump turned the build gate red for every branch (#3795). Treat the
headroom as reserved for the framework, not as room for app code.

The corollary is that the budget cannot catch an app-code regression on its own
any more, so one specific trap is covered by a test:
**do not import `FormsModule` into anything on the eager path.** `vt-login` and
`vt-dialog-host` each render a single uncontrolled text field and used to pull
all ~37 kB of `@angular/forms` for it; both now use plain `[value]` + `(input)`.
`TestEagerBundleComposition` in `tests/core/test_frontend.py` walks the built
bundle's *static* imports from `main.js` and fails if the package reappears.

### Other primitives

`components/icon/` maps names (and backend-supplied emoji) to sanitised inline
SVG, cached per process. `components/context-menu/`, `drop-zone/`,
`skeleton/`, `progress-bar/`, `job-progress/`, `clipboard-copy/` are the small
shared widgets. `toasty-hint/` is one of Toasty's hints (#4680): Toasty and a
speech bubble beside the control to click next, with "Hide this hint" / "Hide
all hints" boxes that `HintsService` keeps in the user's settings. Drop it
under any positioned container; it measures itself against its anchor and
flips the bubble to the other side when it would run off the edge. Inside a
`vt-modal` (however deep) the anchor may be left out: the hint is then about
the dialog box, below it or beside its top corner when the box fills the
window. At most one hint is on screen (#4807): the component shows a hint in
a dialog only while that dialog is the topmost open one (`topOpenModal` in
`modal/modal.component.ts`), and a hint on a view only while no dialog is
open; within a view the host picks one (the Dashboard's `dashboardHint`
ladder, the voting overlay's `hint` input). Its `face`
(happy / sad / surprised, `utils/toasty-faces.ts`) is the same set the toasts
use. [`hints.md`](hints.md) charts every hint: its face, its text and when it fires;
a new hint adds a row there. `directives/no-focus-steal.directive.ts` stops toolbar buttons
next to the Browse canvas from swallowing keyboard focus on mousedown.

Services are root-provided by default; provide one on a component only when
per-instance state or a component-bound lifetime is the point.  The complete
list today: `LabelViewPanelStateService` and `SortRunnerService`
(`label-view`), `PairScopeService` (`label-view`, `find-view`),
`LineTestSessionService` (`find-view`), `BrowseViewportService` /
`BrowseSelectionService` (`browse-view`), and `SiblingLoadService`
(`photo-grid`, `source-photo-modal`).

---

## 9. Styling

[style-guide.md](style-guide.md) is authoritative for design tokens, shared
classes, patterns, and copy style. The structural facts:

- `src/styles.scss` is the global entry and `@use`s the partials in
  `src/scss/`: `_variables` (the `--space-*` / `--font-*` / `--radius-*` /
  color / z-index token set), `_layout`, `_components` (the shared `.btn`,
  `.back-btn`, card, form and modal classes), `_data-table`, `_picker-shared`
  (the `.tab-bar` / `.tab` primitive).
- Everything global is **class-only**, so it applies everywhere without
  bleeding through component encapsulation.
- Component `.scss` files hold only what is genuinely local. A rule that two
  components want belongs in `_components.scss` — that is also how the
  8 kB per-component style budget stays satisfiable.
- Themes (dark / light / high-viz / system) are token swaps driven by
  `ThemeService`; components should read tokens, not hardcode colors.

If your change alters a GUI surface framed by a screenshot in the user docs
or the slides, don't reshoot it: queue the shot in `docs/reshoot-queue/` and
the release run re-renders it. The full rule is "Screenshot reshoots" in
`CLAUDE.md`.

---

## 10. Testing

The suite runs on **Vitest + jsdom** via the `@angular/build:unit-test`
builder. No browser is needed, which is why it works in the cloud container.

```bash
cd frontend && npm run test:ci     # headless, one shot
cd frontend && npm test            # watch
./run-tests.sh frontend            # build:prod + npm audit + Vitest
```

Shared fragments live in `src/app/testing/` and compose freely in a
`providers` array:

- `provideHttpTesting(...interceptors)` — the
  `provideHttpClient` + `provideHttpClientTesting` pair, optionally with
  interceptors under test.
- `provideZoneless()` / `configureZoneless()` — put the `TestBed` under
  zoneless change detection.
- `mocks.ts` — shared service stubs.
- `settle-resource.ts` — `settleResource()` / `settleZoneless(fixture)`, the
  awaits the two subsections below rely on.
- `line-balance.ts` / `line-test.ts` — wire-shaped fixtures for the balance and
  Tune-autopilot payloads.

### The zoneless oracle

A normal spec that pumps `fixture.detectChanges()` by hand **masks** exactly
the staleness bug zoneless introduces: it force-renders regardless of whether
anything was notified. So a spec auditing that property must (1) add
`provideZonelessChangeDetection()`, (2) **remove** the manual pumps — a
leftover pump both re-masks the bug and can race auto-detect into NG0100, (3)
drive updates through the same channel the app uses (write the service
signal, dispatch the bound event), (4) `await settleZoneless(fixture)`, and
(5) assert on rendered DOM, not component fields. The `*.zoneless.spec.ts`
files on the main views are the worked examples.

### `rxResource` timing

The loader is effect-scheduled and promise-based, which changes test timing
even though runtime behaviour is unaffected:

- `fixture.detectChanges()` does **not** issue the GET — call `TestBed.tick()`
  before `httpMock.expectOne(...)`.
- The value commits on a microtask — `await settleResource()` after
  `flush()` (advancing fake timers alone does not commit it).

### Other conventions

`src/test-setup.ts` installs the inert jsdom stubs (`EventSource`,
`ResizeObserver`, `scrollIntoView`, `CSS.escape`, media element methods,
canvas `getContext`) and resets the `TestBed` before each test so a throwing
teardown cannot cascade — that per-test reset *is* the cascade guard. It also
installs the real-XHR guard (`src/app/testing/real-xhr-guard.ts`). `HttpClient`
is provided in root, so a spec without `provideHttpTesting()` gets no missing
provider error, and its requests would go to jsdom's `localhost:3000`. The
guard stops every real `XMLHttpRequest` before it reaches the network and fails
the test that sent it, naming the request.
`vitest.config.ts` deliberately leaves `isolate: false` (the builder's
default): per-file isolation cost 3–4× the runtime and the guard in
`test-setup.ts` covers the same failure mode. No spec uses
`fakeAsync`/`tick`: time is driven with `vi.useFakeTimers()` plus real
macrotask drains.

---

## 11. Build and serve

```bash
cd frontend
npm install
npm start          # dev server on :4200, proxying /api and /static to :5000
npm run build:prod # → ../static/ (index.html, main.js, styles.css)
```

Flask serves the built output from `static/`, whose build artifacts are
gitignored — every deployment builds them. `public/` is copied verbatim into
the output alongside them (favicons, logo); `docs-assets/` is a
symlink to `docs/user/`, so the in-app user guide (rendered with `marked` in
the keyboard-help modal) is the same file the repo ships.

`angular.json` pins the canonical `@angular/build:*` builders (`@angular/build`
is a direct devDependency; the `@angular-devkit/build-angular` alias package is
deliberately not installed) and gives the test target its own
`build:test` configuration, so spec polyfills stay decoupled from production.
Both polyfill arrays are empty and must stay that way — reintroducing
`zone.js` would invalidate every rule in [§5](#5-reactivity-the-zoneless-change-detection-model).
The Angular upgrade runbook is in [`frontend/README.md`](../frontend/README.md).

---

## 12. Invariants

The short list of things that break silently if violated:

- **No `zone.js`.** Not in the app, not in the test build, not in
  `package.json`.
- **Rendered state has a notification path** — signal write, bound event, or
  async-pipe emission. A plain field written from a callback is a stale view.
- **Only `ContextSwitchService` changes the active pair.** Promoting a pair the
  backend has not loaded reopens the 409 cascade.
- **Native `src` URLs go through `mediaUrl()`.** Interceptors do not run for
  `<img>` / `<audio>` / `<video>`.
- **`openapi.json` is regenerated when the backend spec changes**, and the
  generated client is never edited or committed.
- **Hand-written types never duplicate generated ones** — re-export instead.
- **New components are standalone + `OnPush`**, with a `vt-` selector.
- **Build warnings are failures.** Fix the bloat rather than the budget.
- **Desktop only.** No responsive breakpoints, no touch affordances.
