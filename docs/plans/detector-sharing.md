# Design: detector sharing through the Labelset Source family

First-class sharing of detectors between VTSearch users through an external
collaboration service. The service is opaque to this repository: a plugin
written elsewhere talks to it, and this repo holds the contract, the state
model, the routes, the dashboard surfaces and a filesystem-backed test source.

## Background

### What was asked

The request (owner, 2026-11-05) was for six slices: a sharing-provider plugin
contract with an entry-point group; registry state (`remote_ref`, a sync mode,
owner identity, last-synced time, an integrity status) with read-only
enforcement for received detectors at the registry and route layer; share,
refresh, fork, list-received and recipient-search routes that 404 when no
provider is installed; a dashboard split into Saved and Received with a share
dialog, read-only badges, Check for updates, Fork and an integrity warning; a
save-path hook so an owner's saves propagate without a separate export; and a
filesystem test provider with tests. The remote model it must fit: one
container per saved detector holding one record per vote plus small metadata,
a single owner, recipients (users and groups, plus a well-known public group),
server-maintained `created_by` / `last_modified_by` identities on every record
that clients cannot forge, **no server-side read-only for recipients**, owner
saves that rewrite the container in place, and a fork that makes a new
container owned by the forker.

### What the repo already has

Most of the generic half exists, under other names:

- **A bidirectional label-sync plugin family.** `LabelsetSource`
  (`vtscore/labels/sources/base.py`, entry-point group
  `vtscore.labelset_sources`) is a `SyncSource` with `_do_load` /
  `_do_load_full` / `_do_save` / `_do_peek_version`, framework-side field
  normalization, `{detector_id}` / `{detector_name}` template variables, and a
  built-in `server_json_file` source.
- **A per-detector binding and a deployment default.** A registry entry's
  `labelset_source` key (`{"source_name", "field_values"}`) names the source a
  detector syncs with; `default_labelset_source` is the deployment-wide
  fallback; `vtscore/labels/binding.py` resolves the two.
- **The save-path hook, already.** Every detector-file write passes through
  `vtscore.detectors.store._write_detector` / `queue_detector_write`, which call
  `vtscore.labels.sync.note_detector_written`; that schedules a paced,
  debounced, digest-deduplicated push of the file's whole labelset to the bound
  source (#4913), runs it as the detector's creator, and pulls-and-merges on
  load. The request's slice U5 is this engine.
- **Ownership and same-server sharing.** Entries carry `created_by` and
  `readers`; `is_detector_owner` / `can_user_access_detector` gate rename,
  delete, AutoFind and the readers endpoint; the card menu has "Edit access
  list".
- **A frozen-row treatment.** The Detectors card has Drafts and AutoFind tabs,
  and an AutoFind detector renders *frozen*: the editing verbs are hidden
  (`buildDetectorCardMenuItems`, `detector-card.component.html`).
- **An export payload with nothing measured in it.** `labelset_for_export`
  strips thresholds and scores (#5131); `build_detector_meta` is the metadata
  block that travels with a labelset. The "no persisted vectors" rule holds
  for a remote record exactly as it does for a detector file: a record is a
  `LabeledElement` dict.

### Decisions (owner, 2026-11-05)

Three questions decided the shape, and one nuance refined the binding:

- **Sharing is an optional capability of `LabelsetSource`, not a parallel
  plugin kind.** The deciding argument is the push engine: an owner's paced
  push is exactly how updates reach recipients, and a linked (received)
  detector must never push. One engine made *role-aware* gives both; a second
  plugin kind would mean a second engine with the same pacing, dedup and
  as-the-creator rules to keep in step.
- **Activation is an explicit server setting, default off.** Exporters and
  importers are per-action menus where "show everything installed" is right.
  Sharing is an ambient, deployment-level integration: it changes the
  Detectors card, adds push behaviour, and implies a backing service with its
  own auth and data residency. Ambient integrations are configuration, not an
  inference from what happens to be importable. The built-in test source is
  discovered like any other built-in, so "the only installed source is active"
  would switch sharing on everywhere.
- **Received detectors are badged inside the existing tabs.** No new tab: a
  linked detector shows in Drafts or AutoFind with a Received badge and the
  frozen treatment, and the containers available to the user live in their
  own small list, not on the card.
- **The per-detector binding stays, and records which source a container
  belongs to.** An entry's `labelset_source` names the source its
  `remote_ref` is for. The setting governs only the ambient surfaces
  (listings, the share dialog, recipient search) and *new* saves. An entry
  that references a source other than the active one (after a deployment
  change, say) is treated as **local-only with a note**: no push, no refresh,
  the sharing actions disabled with the reason. Never guess a different
  source.

### Where this plan departs from the request

- **U5 is not new code.** The hook exists; the work is to make it role-aware
  (an owner push goes to the container, a linked detector never pushes or
  pull-merges) and to record a failed push on the entry.
- **Fork is create, with provenance.** The request lists `fork(remote_ref)`
  on the provider. The copy a user forks must be the *local* one: after a
  refresh where they chose to skip foreign records, the local file is the
  trusted labelset and the container is the tampered one. So the core forks by
  writing a new container from the local labelset through the create hook and
  records `forked_from` itself. No provider fork hook.
- **"Retraining" is allowed; changing the labelset is not.** A linked
  detector still trains its head in RAM from its labelset, or it could not
  sort, find or run as AutoFind. What is refused is any change to its labelset
  or identity: votes, label edits and imports, rename, being a combine
  target, a backup-source pull that would write the file. Derived numbers
  (threshold, balance, test verdict, the AutoFind flag) stay writable.
- **A receive step is needed.** `list_received` names containers; nothing in
  the request turns one into a local detector. `receive(remote_ref)` creates
  the linked entry and file from a refresh.
- **Identity is the source's.** VTSearch usernames and the service's
  identities are different namespaces. The source resolves the caller through
  `get_current_user()` (the push engine already runs as the detector's
  creator under `thread_user`), owns the credential mapping, and reports
  identities as opaque strings the core stores and displays verbatim.
- **Deletions are a tamper vector too.** A recipient who deletes a record
  leaves no `last_modified_by` to inspect. So an owner save writes a manifest
  (record count and digest) into the container's metadata, and a refresh
  reports the manifest's own `last_modified_by`; a manifest not written by the
  owner, or a count that disagrees with the records, is a foreign
  modification.
- **The in-app `readers` list is untouched.** It is same-server visibility.
  A linked detector can be given readers like any other entry; they see a
  read-only row.

## The design

### Vocabulary and the entry shape

| Term | Meaning |
|---|---|
| **container** | The remote object one saved detector maps to: records plus metadata, one owner. Named by an opaque `remote_ref` string. |
| **owner** | The identity, as the source reports it, that saved the container. Local ownership stays `created_by`. |
| **sync mode** | `local` (the default and the absent value), `linked` (a received, read-only mirror) or `forked` (`local` in every rule, with `forked_from` provenance). |
| **active sharing source** | The `LabelsetSource` the `detector_sharing_source` server setting names. `null` means sharing is off. |

The entry shape shipped in #5575 and is documented in
`vtscore/docs/packages/detectors.md` ("Sharing state"): a top-level
`sync_mode`, the container's state inside the `labelset_source` binding
(`remote_ref`, `remote_owner`, `last_synced`, `integrity`, `sync_error`; the
setting's fixed field values are laid in at call time, never stored), and
`forked_from`. `GET /api/detectors/registry` lists `sync_mode`, `sharing`
(the binding without its field values), `sharing_note` and `forked_from`.

### The sharing contract

Shipped in #5574; the permanent reference is "The sharing contract" in
`vtscore/docs/extending/labelset-sources.md`. What the later slices build on:
the core calls the `share_*` wrappers with `share_payload(detector_data)`'s
`(labelset, meta)` pair (`meta` is a `ShareMeta`: name, `detector_meta`,
manifest), reads the source through `active_sharing_source()`, and checks a
refresh's integrity with `build_manifest(container.as_labelset())` against
`container.manifest` and `manifest_modified_by`. An unsupported optional
operation raises `SharingUnsupported`, which a route answers with 501.

### Roles in the sync engine

`vtscore.labels.sync` keeps one slot per detector and one pacing policy. What
changes is what a push does, by role:

| Entry | On a detector-file write | On load | Manual sync route |
|---|---|---|---|
| plain binding (today) | paced `_do_save` | pull, file wins | pull, source wins |
| owner (binding has `remote_ref`, `sync_mode` local/forked, active source) | paced `_do_share_save` with the manifest, **no pull-first merge**; failure recorded as `sync_error` | nothing | 409 (use refresh semantics: nothing to pull into an owner) |
| linked | **nothing** | **nothing** | 409 `detector_read_only` |
| binding names a source that is not the active one | nothing, with a note | nothing | 409 with the note |

The pull-first merge that protects a plain binding (a backup the server lost)
is the wrong policy for an owner: merging the container into the owner's file
would import exactly the foreign records tamper detection exists to catch.
The owner's file is authoritative.

### Integrity and the read-only guarantee

Shipped in #5575: `compute_integrity`, the `abort` / `import` / `skip`
refresh policies, `receive_container`, `refresh_linked_detector` and
`fork_detector` are in `vtscore/docs/packages/labels.md` ("Receive, refresh,
fork"); the store guard (`DetectorReadOnlyError`, `allow_linked_write`) is in
`vtscore/docs/packages/detectors.md` ("The read-only guarantee"), and every
route that changes a labelset or an identity answers 409
`detector_read_only` before any side effect
(`vtsearch/routes/_read_only.py`). What the later slices build on: the
routes map `SharingUnavailable` to the note's 409, `ForeignModifications`
from a receive to the integrity dialog, and `LookupError` to 404; the
dashboard offers a fork on the `detector_read_only` code.

### Routes

New, under a `detectors_sharing_bp`; all but the first 404
`sharing_disabled` when the setting is null:

| Route | Does |
|---|---|
| `GET /api/detectors/sharing` | Capability: enabled, the source's name and icon, `supports_recipient_search`. 200 always |
| `GET /api/detectors/sharing/received` | `list_received`, each with the local `detector_id` when already received |
| `POST /api/detectors/sharing/receive` | `{remote_ref, foreign?}` → a new linked entry and file |
| `POST /api/detectors/sharing/search-recipients` | `{query}` → suggestions; 501 when unsupported |
| `GET /api/detectors/registry/<id>/sharing` | The binding's sharing state, recipients and public flag |
| `POST /api/detectors/registry/<id>/share` | `{add, remove, public}`; creates the container on first use (replaces a plain binding, and says so) |
| `DELETE /api/detectors/registry/<id>/share` | Archive and unbind |
| `POST /api/detectors/registry/<id>/refresh` | `{foreign: abort\|import\|skip}`; linked only |
| `POST /api/detectors/registry/<id>/fork` | `{name?}` → 201, a `forked` entry owned by the caller |

### Dashboard

- A **Received** badge on linked rows, with the frozen treatment, plus "Check
  for updates" and "Fork" in the menu. Delete stays and removes the local copy
  only. Move to AutoFind stays: the flag is local and the guard allows it.
- A **Shared** badge on owned rows with a container, "Share…" in the menu
  (replacing "Auto-export…" for that binding), and a "Remote sync failed"
  link where the auto-export warning sits today.
- A note, and disabled actions, on a row whose source is not the active one.
- A **Received…** button in the card header, shown only when sharing is on,
  opening the list of containers available to the user with an Add action.
- Voting and label editing on a linked detector are disabled up front from
  `sync_mode`, with a fork affordance; the 409 is the backstop and shows a
  toast with a Fork action.

### Out of scope

Unifying the in-app `readers` list with remote recipients; sharing datasets;
dual bindings (a shared detector that also pushes to a plain backup location,
rejected for now because one binding per detector is what the engine and the
dashboard reason about); a settings-modal editor for the server setting.

## Dependency graph

Which issues can run at once. An arrow means "needs merged first".

```
#5574  contract, shared_folder, setting
  │
  ▼
#5575  entry shape, write guard, receive / refresh / fork
  │
  ├────────────────────┐
  ▼                    ▼
#5576  sync roles    #5577  routes
                       │
                       ├────────────────────┐
                       ▼                    ▼
                     #5578  dashboard     #5579  dialogs
                       │                    │
                       └─────────┬──────────┘
                                 ▼
                              #5580  user guide
```

| Wave | Run in parallel | Waits for |
|---|---|---|
| 1 | #5574 | nothing |
| 2 | #5575 | #5574 |
| 3 | #5576 and #5577 | #5575 |
| 4 | #5578 and #5579 | #5577 |
| 5 | #5580 | #5578 and #5579 |

- **#5576 and #5577 are independent in code.** The sync engine
  (`vtscore/labels/sync.py`, the labels package doc) and the routes (a new
  blueprint, schemas, the OpenAPI snapshot, `docs/api/`) share no file. Until
  both are on `dev`, a detector shared through the routes pushes through the
  engine's plain path, where a sharing-only source raises `NotImplementedError`
  into the engine's log; harmless, and gone once #5576 lands.
- **#5578 and #5579 are independent in code**, but both regenerate the API
  client from #5577's snapshot, so neither starts before #5577 merges. They
  meet in two places: the "Share…" verb #5578 adds is what #5579's dialog
  answers (a no-op until both are in), and the integrity dialog #5578 builds
  is what #5579's receive flow reuses. Whichever lands second wires the pair.
- **#5580 captures screenshots of the finished surfaces**, so it waits for
  both.

## Open work

<!-- item-sep -->

- [ ] #5574 — Sharing as an optional capability of `LabelsetSource`: the contract, the `shared_folder` built-in, the `detector_sharing_source` setting, and conformance tests — Opus

<!-- item-sep -->

- [ ] #5575 — Registry sharing state, the linked-detector write guard, and the receive, refresh and fork operations — Opus

<!-- item-sep -->

- [ ] #5576 — Role-aware label sync: owner pushes rewrite the container without merging, linked detectors never push or pull, failures land on the entry — Opus

<!-- item-sep -->

- [ ] #5577 — Detector sharing routes: capability, received listing, receive, share and unshare, refresh, fork, recipient search — Sonnet

<!-- item-sep -->

- [ ] #5578 — Dashboard: Received and Shared badges, read-only gating with fork-instead, Check for updates with the integrity dialog, Fork — Opus

<!-- item-sep -->

- [ ] #5579 — Share dialog and the Received list modal — Sonnet

<!-- item-sep -->

- [ ] #5580 — User guide: share a detector, receive one, fork one — Sonnet

<!-- item-sep -->
