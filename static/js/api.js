/**
 * Centralised fetch helpers for every VTSearch API endpoint.
 *
 * Each function returns a parsed JSON body (or throws).  Callers can
 * destructure or chain as needed.
 */

export async function fetchJSON(url, opts) {
  const res = await fetch(url, opts);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    const err = new Error(body.error || `HTTP ${res.status}`);
    err.status = res.status;
    err.body = body;
    throw err;
  }
  return res.json();
}

export async function postJSON(url, data) {
  return fetchJSON(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
}

export async function putJSON(url, data) {
  return fetchJSON(url, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
}

// ---- Specific endpoints ------------------------------------------------

export const getMedias        = ()           => fetchJSON("/api/medias");
export const getVotes         = ()           => fetchJSON("/api/votes");
export const getInclusion     = ()           => fetchJSON("/api/inclusion");
export const postVote         = (id, vote)   => postJSON(`/api/medias/${id}/vote`, { vote });
export const postInclusion    = (inclusion)  => postJSON("/api/inclusion", { inclusion });

export const getDatasetStatus = ()           => fetchJSON("/api/dataset/status");
export const getDemoList      = ()           => fetchJSON("/api/dataset/demo-list");
export const getAllImporters   = ()           => fetchJSON("/api/dataset/all-importers");

export const getSettings      = ()           => fetchJSON("/api/settings");
export const putSettings      = (data)       => putJSON("/api/settings", data);
export const getDefaults      = ()           => fetchJSON("/api/settings/defaults");

export const postTextSort     = (text)       => postJSON("/api/sort", { text });
export const postLearnedSort  = (signal)     => fetchJSON("/api/learned-sort", { method: "POST", headers: { "Content-Type": "application/json" }, signal });
export const postDetectorSort = (detector)   => postJSON("/api/detector-sort", { detector });
export const getSortProgress  = ()           => fetchJSON("/api/sort/progress");

export const getExporters     = ()           => fetchJSON("/api/exporters");
export const postExport       = (data)       => postJSON("/api/exporters/export", data);

export const getLabelImporters      = () => fetchJSON("/api/label-importers");
export const getProcessorImporters  = () => fetchJSON("/api/processor-importers");
export const getAutorunDetectors    = () => fetchJSON("/api/autorun-detectors");

export const getMediaTypes    = ()           => fetchJSON("/api/media-types");
export const getLabelingStatus = ()          => fetchJSON("/api/labeling-status").catch(() => null);

export const getServerMediaFiles    = () => fetchJSON("/api/server-media-files");
export const getServerDetectorFiles = () => fetchJSON("/api/detector/server-files");

export const postExampleSort  = (formData) =>
  fetchJSON("/api/example-sort", { method: "POST", body: formData });

export const postExampleSortServer = (filename) =>
  postJSON("/api/example-sort-server", { filename });

export const postAutoDetect   = (body)       => postJSON("/api/auto-detect", body);
export const postFind         = (body)       => postJSON("/api/find", body);

export const postDiversityTreeNext = (body)  => postJSON("/api/diversity-tree/next", body || {});

export const getDatasetsRegistry   = ()      => fetchJSON("/api/datasets/registry");
export const getModelsRegistry     = ()      => fetchJSON("/api/models/registry");

export const postTextsortSuggestion = (text) =>
  postJSON("/api/textsort-suggestions", { text }).catch(() => {});

export const getTrainableModels    = ()      => fetchJSON("/api/trainable-models");
