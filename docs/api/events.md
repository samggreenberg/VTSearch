# Progress events (SSE)

VTSearch streams progress for every long-running operation through a
single [Server-Sent Events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events)
endpoint. Replaces the per-tracker REST polling endpoints
(`/api/dataset/progress`, `/api/sort/progress`, `/api/find/progress`,
`/api/dataset/loading-tasks`, `/api/detectors/loading-tasks`,
`/api/eval/voting-iterations`).

## Endpoint

```
GET /api/events
```

`Content-Type: text/event-stream`. Connect with `new EventSource('/api/events')`.

The first frame on every channel is the **current snapshot** of that
tracker, so clients do not need a separate REST bootstrap call.

## Channels

| Event name | Payload | Source |
|---|---|---|
| `dataset` | progress object | singleton `dataset_progress` tracker (legacy single-op fallback) |
| `loading-tasks` | array of task objects | `loading_tasks` (parallel dataset loads and staging imports; a staging task carries `staging_result`) |
| `detector-loading-tasks` | array of task objects | `detector_loading_tasks` |
| `sort` | progress object | `sort_progress` (text sort) |
| `find` | progress object | `find_progress` (multi-dataset Find) |
| `eval` | progress object | `eval_progress` (train-and-score) |
| `notification` | notification object | `notify()` — one-off messages from server-side code; rendered as toasts |
| `heartbeat` | `{ "ts": <unix seconds> }` | periodic liveness ping (every ~5s) |

Every channel except `notification` carries **state**: a snapshot sent on
connect and re-sent on every heartbeat, so a dropped frame heals itself.
`notification` carries **events** — see [Notification object
shape](#notification-object-shape) for what that costs.

### Progress object shape

```json
{
  "status": "loading",
  "message": "Embedding medias…",
  "current": 50,
  "total": 500,
  "step": 3,
  "total_steps": 4,
  "overall": 0.625,
  "eta_seconds": 30.0,
  "error": null
}
```

Optional fields (`step`, `total_steps`, `overall`, `eta_seconds`, `error`,
`staging_result`) are included only for trackers that declare them.

When a tracker reports a `step`/`total_steps` structure, it also exposes a
single whole-job completion fraction in `overall` (0..1) and an `eta_seconds`
estimate for the *entire* job. `overall` advances once across all phases
(e.g. download → load model → embed → finalize) instead of resetting at each
phase, so consumers should prefer it for the progress bar and fall back to
`current`/`total` only when `overall` is `null` (single-phase operations).

### Task object shape (`loading-tasks` / `detector-loading-tasks`)

```json
{
  "task_id": "task_abc",
  "name": "ESC-50",
  "status": "loading",
  "message": "Embedding…",
  "current": 50,
  "total": 500,
  "error": null,
  "created_at": 1731000000.123,
  "dataset_id": "esc50",
  "media_type": "audio",
  "embedder": "clap"
}
```

`dataset_id`, `detector_id`, `media_type`, `embedder` are only present
when the task carries that information.

Some tasks add a terminal result payload, `null` until they finish:
combine-datasets staging publishes `staging_result`, and a labelset-media
ingest (see [detectors.md](detectors.md) and [io.md](io.md)) publishes
`ingest_result` — `{"ingested", "applied", "unresolved", "failed"}`, with only
`ingested` present on the detector-import path.

### Notification object shape

```json
{
  "id": "note_4eb45583_7",
  "level": "warning",
  "message": "Skipped 3 unreadable files",
  "detail": "page_2.pdf, page_9.pdf, notes.pdf",
  "source": "Server Folder",
  "timestamp": 1731000000.123
}
```

A one-off message from server-side code — typically a plugin that hit a
recoverable problem and chose to carry on rather than fail the whole
operation. Published with `notify()`
(`vtscore/concurrency/notifications.py`); the Angular client turns each one
into a toast via `ToastService`. `level` is one of `info`, `success`,
`warning`, `error`; the first two auto-dismiss, the last two stay until the
user closes them. `detail` and `source` may be `null`.

Unlike every other channel this one has no snapshot, which has three
consequences worth designing around:

- **No bootstrap.** It is absent from the connect-time snapshot.
- **No replay.** A notification published while nobody is connected is gone;
  a client connecting a second later does not receive it.
- **No heal.** Each client's queue is bounded, and a notification dropped
  when that queue is full is not re-sent by the heartbeat the way a tracker
  snapshot is. The backend logs the drop.

Every open client receives every notification, including a second browser
tab — there is no per-session routing.

## Frame format

```
event: dataset
data: {"status":"loading",...}

```

A `heartbeat` event arrives every ~5 seconds so connections survive idle
proxies **and** so the client has a continuous "backend is alive" signal: the
Angular client treats every frame (heartbeat or real progress) as proof of
life and only declares the backend offline when the stream goes silent. It is
a real named event rather than an SSE comment (`: heartbeat`) precisely because
comments are invisible to the browser's `EventSource` API. Each heartbeat tick
also re-emits **every** channel's current snapshot. For `loading-tasks` and
`detector-loading-tasks` that is what makes finished tasks vanish from the UI
once they pass their stale-prune window, without a server-side timer. For the
single-tracker channels (`dataset`, `sort`, `find`, `eval`) it is a self-heal:
each client's queue is bounded and drops frames when the client stalls, so a
channel's single terminal `idle`/`error` frame can be lost and would otherwise
leave a progress bar stuck at its last percentage until the next operation
fired that channel. Re-emitting the (tiny) snapshots makes every channel
eventually consistent after any drop.

Between heartbeats an idle stream also writes a bare SSE comment (`: ka`)
roughly every second. It is deliberately a comment — invisible to
`EventSource`, so it does not feed the client's liveness signal — and exists
only to probe the socket: a client that vanished abruptly (page reload) is
detected on the first failed write, releasing its connection slot in ~1s
instead of a full heartbeat period.

## Example

```ts
const es = new EventSource('/api/events');
es.addEventListener('dataset', (e) => {
  const { current, total, message } = JSON.parse(e.data);
  setProgress(current / total, message);
});
es.addEventListener('loading-tasks', (e) => {
  setActiveTasks(JSON.parse(e.data));
});
es.addEventListener('notification', (e) => {
  const { level, message, detail } = JSON.parse(e.data);
  showToast(level, message, detail);
});
```

The browser's `EventSource` reconnects automatically on transient
network failures. The Angular client (`ProgressEventsService`) also
schedules a manual reconnect 2 s after `readyState === CLOSED` for the
rare case the server explicitly closes the stream.
