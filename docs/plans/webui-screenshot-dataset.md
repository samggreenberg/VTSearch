# WebUI web-page screenshot demo dataset (deferred)

## Background

VTSearch gained three born-digital / digitally-native image demo datasets to
improve and measure the image embedder on screenshots rather than natural
photos: **Enrico** (mobile UI screenshots by screen function), **RICO-Screen2Words**
(mobile UI screenshots by Google Play app category), and **RVL-CDIP** (scanned
document images by type). The shared HuggingFace-parquet image helper lives in
`vtscore/datasets/downloader/_hf_parquet.py`.

The originally-scoped fourth dataset, **WebUI** (≈400K rendered *web-page*
screenshots, CHI 2023, `js0nwu/webui` / HF org `biglab`), is **not shipped** —
it is the only web-page (vs. mobile/document) screenshot source considered, so
it's worth finishing if the blockers below are resolved. This file tracks that
remaining work.

<!-- item-sep -->

- **WebUI web-page screenshot demo** — Add WebUI as an image demo so the
  screenshot coverage includes desktop/tablet/phone *web pages*, not just mobile
  app UIs. Two hard blockers must be solved first, and a product decision made:

  - **Extraction: needs multi-part (spanned) zip support.** The smallest usable
    split is `biglab/webui-7kbal` (~5.7 GB; `biglab/webui-7k` is ~8.4 GB). Each
    is a split LFS archive (`balanced_7k.zip.001` + `.zip.002`) of loose
    per-sample directories of **`.webp`** screenshots (one viewport + one
    full-page per simulated device: 4 desktop resolutions, iPad, iPhone). Python
    `zipfile` cannot open spanned archives and `py7zr` does not handle
    multi-volume *zip* (only `.7z`). Options: (a) add a `7z`/`7za` CLI extraction
    path to `vtscore/datasets/downloader/core._extract_archive` (new external
    system dependency — gate it behind an actionable error like the `py7zr`
    branch), or (b) find/ build a parquet mirror of a WebUI subset and reuse
    `_hf_parquet` (none known to exist; the HF viewer is disabled on these repos).
    Pull via `huggingface_hub.snapshot_download(repo_id="biglab/webui-7kbal",
    repo_type="dataset", allow_patterns=["*.zip.*", "*.json"])`.

  - **Labels: none are distributed.** The released `biglab/webui-*` data carries
    **no per-screenshot category/genre label**. The paper's 20-class "screen
    classification" labels are Enrico-derived *pseudo-labels* produced during
    their experiments, not shipped. So a category-based demo/eval must
    **synthesize** a per-image label. The only real per-image signals present
    are: **device form-factor** (desktop / tablet / phone, from the screenshot
    filename — e.g. `iPhone-13 Pro-screenshot.webp`, `default_1920-1080-...`),
    and the **source domain** (`url.txt` per sample, high-cardinality). Decide:
    ship as a search-only demo with device form-factor categories (a real,
    weak-but-honest 3-way label good enough for a small eval), or a single
    "webpage" bucket for the vote-to-train flow with no eval. Do **not** invent
    the 20 UI-function classes here — those labels are not in the data.

  - **License note.** HF tag is "other"; governing terms are CMU's
    `COPYRIGHT.txt` (screenshots may contain copyrighted work; user assumes
    liability). Freely downloadable per-user from HF (no gating), but do not
    redistribute the screenshots ourselves — the download-on-demand model (as
    with the other demos) is the intended path.

<!-- item-sep -->
