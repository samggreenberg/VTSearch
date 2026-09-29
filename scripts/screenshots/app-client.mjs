/**
 * A client for a running VTSearch app, shared by the screenshot harnesses.
 *
 * Both the user guide's shots (`capture.ts`, `ensure-fixtures.mjs`) and the
 * slide deck's (`slides/figs/src/shoot-ui-figs.mjs`) drive one running app
 * through the same few API calls: import a corpus if it is absent, create and
 * load a detector, reset its votes to a fixed baseline. Each binds this client
 * to its own worked example — `smiley-example.mjs` for the guide,
 * `book-example.mjs` for the deck.
 */

/**
 * A small client for the running app, plus the idempotent fixture steps the
 * harnesses need. *log* is the caller's logger, so messages say whose they are.
 *
 * *example* binds it to one worked example: `corpusPath(name)` returns the
 * server path of that example's corpus *name*, building it first if need be,
 * and `query` is the text a new detector is seeded with.
 */
export function appClient(app, log = () => {}, { corpusPath, query }) {
  async function api(path, { method = 'GET', body, dataset, detector } = {}) {
    const headers = { 'content-type': 'application/json' };
    if (dataset) headers['X-Dataset-Id'] = dataset;
    if (detector) headers['X-Detector-Id'] = detector;
    const r = await fetch(app + path, {
      method,
      headers,
      body: body === undefined ? undefined : JSON.stringify(body),
    });
    if (!r.ok) throw new Error(`${method} ${path} -> ${r.status} ${await r.text()}`);
    return r.json();
  }

  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  async function waitFor(what, predicate, timeoutMs = 1_800_000) {
    const until = Date.now() + timeoutMs;
    while (Date.now() < until) {
      const hit = await predicate();
      if (hit) return hit;
      await sleep(2000);
    }
    throw new Error(`timed out waiting for ${what}`);
  }

  const datasets = async () => (await api('/api/datasets/registry')).datasets || [];
  const detectors = async () => (await api('/api/detectors/registry')).detectors || [];
  const named = (rows, name) => rows.find((r) => r.name === name);

  /** Load a registered dataset that is on disk but not in memory. */
  async function loadDataset(row) {
    if (row.loaded) return;
    await api(`/api/datasets/registry/${row.id}/load`, { method: 'POST' });
    await waitFor(`dataset ${row.name} to load`, async () => named(await datasets(), row.name)?.loaded);
  }

  /**
   * Import corpus *name* with *embedder* unless it is already registered.
   *
   * Registered is not loaded: a fresh import leaves the dataset in memory, but a
   * re-run against a restarted app finds it on disk and unloaded, and every call
   * after this one 409s with `dataset_not_loaded`. Idempotent means idempotent
   * across restarts too.
   */
  async function ensureDataset(name, embedder, extra = []) {
    const existing = named(await datasets(), name);
    if (existing) {
      log(`dataset ${name} exists (${existing.num_items} items)`);
      await loadDataset(existing);
      return existing;
    }
    const path = corpusPath(name);
    log(`importing ${name} (${embedder}) — embedding takes a while on CPU`);
    await api('/api/dataset/import/server_folder', {
      method: 'POST',
      body: {
        path,
        media_type: 'image',
        recursive: 'true',
        reference_files: 'true',
        dataset_name: name,
        embedder,
        // The optional region / instance embedders, as the Add Dataset
        // dialog's Advanced section sends them: the whole trio, primary first.
        ...(extra.length ? { embedders: [embedder, ...extra] } : {}),
      },
    });
    const row = await waitFor(`dataset ${name}`, async () => named(await datasets(), name));
    log(`imported ${name} (${row.num_items} items)`);
    return row;
  }

  /**
   * Create (if absent) and load a trainable image detector on *dataset*, seeded
   * by *text*. *embedderType* locks it to one of the dataset's embedder types
   * (`patch_semantic` for region voting); a detector already locked to a
   * different type is dropped and made again.
   */
  async function ensureDetector(name, dataset, text = query, embedderType = '') {
    let existing = named(await detectors(), name);
    if (existing && embedderType && existing.embedder_type !== embedderType) {
      await dropDetectors(name);
      existing = undefined;
    }
    const row =
      existing ||
      (
        await api('/api/detectors/registry', {
          method: 'POST',
          dataset: dataset.id,
          body: {
            name,
            media_type: 'image',
            text_query: text,
            trainable: true,
            ...(embedderType ? { embedder_type: embedderType } : {}),
          },
        })
      ).detector;
    await api('/api/detectors/registry/load', {
      method: 'POST',
      dataset: dataset.id,
      body: { detector_id: row.id },
    });
    await waitFor(`detector ${name} to load`, async () => named(await detectors(), name)?.loaded);
    return row;
  }

  /** Unregister every dataset called one of *names* (and its pickle). */
  async function dropDatasets(...names) {
    for (const row of await datasets()) {
      if (!names.includes(row.name)) continue;
      await api(`/api/datasets/registry/${row.id}`, { method: 'DELETE' });
      log(`removed the previous ${row.name} dataset`);
    }
  }

  /** Delete every detector called one of *names*. */
  async function dropDetectors(...names) {
    for (const row of await detectors()) {
      if (!names.includes(row.name)) continue;
      await api(`/api/detectors/registry/${row.id}`, { method: 'DELETE' });
      log(`removed the previous ${row.name} detector`);
    }
  }

  /** Every item of *dataset* as `{id, filename}`. */
  async function mediaIndex(dataset, detector) {
    const ids = (await api('/api/medias/ids', { dataset: dataset.id, detector: detector.id })).map(
      (m) => m.id
    );
    return api('/api/medias/batch', {
      method: 'POST',
      dataset: dataset.id,
      detector: detector.id,
      body: { ids },
    });
  }

  /** Cast one vote, riding out the 409 a detector gives while it settles. */
  async function vote(dataset, detector, id, target) {
    for (let attempt = 0; attempt < 20; attempt++) {
      const r = await fetch(`${app}/api/medias/${id}/vote`, {
        method: 'POST',
        headers: {
          'content-type': 'application/json',
          'X-Dataset-Id': dataset.id,
          'X-Detector-Id': detector.id,
        },
        body: JSON.stringify({ target }),
      });
      if (r.ok) return;
      // 409 is "the detector is still settling"; anything else is a real error.
      if (r.status !== 409) throw new Error(`vote ${id} -> ${r.status}`);
      await sleep(1000);
    }
    throw new Error(`vote ${id} still 409 after retries`);
  }

  /**
   * Put exactly *good* and *bad* (file names) on the detector, and nothing else.
   *
   * Clears first, so a re-run against a detector that later recipes voted on
   * starts from the same baseline instead of drifting.
   */
  async function setVotes(dataset, detector, { good, bad }) {
    const meta = await mediaIndex(dataset, detector);
    const byName = Object.fromEntries(meta.map((m) => [m.filename, m.id]));
    const ids = (names) =>
      names.map((name) => {
        const id = byName[name];
        if (id === undefined) throw new Error(`${name} is not in the ${dataset.name} corpus`);
        return id;
      });
    await api('/api/votes/clear', { method: 'POST', dataset: dataset.id, detector: detector.id });
    for (const id of ids(good)) await vote(dataset, detector, id, 'good');
    for (const id of ids(bad)) await vote(dataset, detector, id, 'bad');
    await sleep(3000);
    log(`${detector.name ?? 'detector'}: ${good.length} good / ${bad.length} bad`);
    return meta;
  }

  return {
    api,
    sleep,
    waitFor,
    datasets,
    detectors,
    named,
    loadDataset,
    ensureDataset,
    ensureDetector,
    dropDatasets,
    dropDetectors,
    mediaIndex,
    vote,
    setVotes,
  };
}
