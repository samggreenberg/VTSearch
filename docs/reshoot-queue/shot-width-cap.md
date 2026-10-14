# Docs shots written at most 1440 px wide

`capture.ts` now downscales any shot wider than `MAX_WIDTH` (1440 px, 2× the
guide's 720 px embed) before encoding, so every full-window shot shrinks from
2880 × 1800 to 1440 × 900 at the next refresh. The drain re-renders the whole
docs set; these are only the most visible.

- `dashboard-loaded` — the README's hero image, and the pattern for every full-window shot
