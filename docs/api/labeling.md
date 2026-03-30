# Labeling & Diversity

[← Back to API index](../API.md)

---

## Inclusion & Thresholds

### Get / set inclusion

```
GET /api/inclusion
```

→ `{"inclusion": 0}`

```
POST /api/inclusion
```

**Body:** `{"inclusion": 3}`

Value is clamped to the range -10 to +10.

→ `{"inclusion": 3}`

### Get / set safe thresholds

```
GET /api/safe-thresholds
```

→ `{"safe_thresholds": false}`

```
POST /api/safe-thresholds
```

**Body:** `{"safe_thresholds": true}`

→ `{"safe_thresholds": true}`

---

## Labeling Progress

### Analyze progress

```
POST /api/labeling-progress
```

Requires at least one good vote, one bad vote, and label history.

→ Analysis object with progress metrics (structure depends on internal
implementation).

### Labeling status indicators

```
GET /api/labeling-status
```

→ ```json
{
  "smart": {"status": "green"},
  "stable": {"status": "yellow"},
  "span": {"status": "red"}
}
```

Each metric has a `status` of `"red"`, `"yellow"`, or `"green"`.

### Indicator score history

```
GET /api/indicator-score-history
```

**Query params:** `metric` — one of `"smart"`, `"stable"`, `"diverse"`.

→ `{"metric": "smart", "history": [...]}`

Returns cached per-step indicator data (computed during labeling-status
polling).

### Evaluate metric (train-and-score)

```
POST /api/eval/train-and-score
```

**Body:** `{"metric": "smart"}` (or `"stable"` / `"diverse"`)

→ `{"error_cost": [...]}` (smart), `{"stability": [...]}` (stable), or
`{"diversity": [...]}` (diverse).

### Evaluation progress

```
GET /api/eval/voting-iterations
```

→ `{"progress": 5, "total": 10, "done": false}`

---

## Diversity Tree

### Get next diverse sample

```
GET /api/diversity-tree/next
POST /api/diversity-tree/next
```

POST accepts an optional body with sort scores to influence selection:

**Body:** `{"scores": {"0": 0.9, "1": 0.2}}`

→ `{"id": 42, "diversity_level": 3, "exhausted": false}`

`id` is `null` when the tree is not built or exhausted. `diversity_level` is the
number of consecutive seen nodes in BFS order (0 when nothing is labeled, up to
the total number of tree nodes when fully covered). `exhausted` is `true` when
every node has been seen.
