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
| `dataset` | progress object | singleton `dataset_progress` tracker (staging, embedding) |
| `loading-tasks` | array of task objects | `loading_tasks` (parallel dataset loads) |
| `detector-loading-tasks` | array of task objects | `detector_loading_tasks` |
| `sort` | progress object | `sort_progress` (text sort) |
| `find` | progress object | `find_progress` (multi-dataset Find) |
| `eval` | progress object | `eval_progress` (train-and-score) |

### Progress object shape

```json
{
  "status": "loading",
  "message": "Embedding medias…",
  "current": 50,
  "total": 500,
  "step": 1,
  "total_steps": 3,
  "error": null
}
```

Optional fields (`step`, `total_steps`, `error`, `staging_result`) are
included only for trackers that declare them.

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

## Frame format

```
event: dataset
data: {"status":"loading",...}

```

Comment lines (`: heartbeat`) arrive every ~5 seconds so connections
survive idle proxies. They also re-emit the `loading-tasks` and
`detector-loading-tasks` channels so finished tasks vanish from the UI
once they pass their stale-prune window without a server-side timer.

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
```

The browser's `EventSource` reconnects automatically on transient
network failures. The Angular client (`ProgressEventsService`) also
schedules a manual reconnect 2 s after `readyState === CLOSED` for the
rare case the server explicitly closes the stream.
