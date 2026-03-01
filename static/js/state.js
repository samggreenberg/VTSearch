/**
 * Centralized state management for VTSearch.
 *
 * All mutable application state lives here behind getter/setter helpers so
 * that future migrations (e.g. to a reactive store) are localised to this
 * one module.
 */

const State = {
  // ---- Media & voting --------------------------------------------------
  medias: [],
  votes: { good: [], bad: [], click_times: {}, learned_scores: {} },
  selected: null,

  // ---- Sort ------------------------------------------------------------
  sortOrder: null,       // null = default, or [{id, score}, ...]
  sortMode: "text",      // "text" | "learned" | "load"
  selectMode: "top",     // "top" | "hard" | "new"
  threshold: null,
  sortTimer: null,
  loadedDetector: null,

  // ---- UI flags --------------------------------------------------------
  datasetLoaded: false,
  currentView: "welcome", // "welcome" | "dashboard" | "labeling"
  isVoting: false,
  swipeAnimation: true,
  showThumbnailsLeft: false,
  showThumbnailsRight: true,
  labelSortMode: "time-desc",

  // ---- Audio -----------------------------------------------------------
  audioVolume: 1.0,
  volumeSaveTimer: null,
  waveformAudioCtx: null,

  // ---- Progress / polling ----------------------------------------------
  progressTimer: null,
  progressEtaState: null,
  sortProgressTimer: null,
  sortEtaState: null,

  // ---- AbortControllers & debounce timers ------------------------------
  learnedSortController: null,
  paragraphController: null,
  learnedSortDebounce: null,

  // ---- Combine-datasets staging ----------------------------------------
  _combineState: null,

  // ---- Media-type metadata (keyed by type_id) --------------------------
  mediaTypesMap: {},

  // ---- Autorun / detector bookkeeping ----------------------------------
  autorunDetectors: [],
  favoriteDetectors: [],

  // ---- Dashboard -------------------------------------------------------
  dashSelectedDataset: null,
  dashSelectedDetector: null,
  dashSelectedDatasetIds: [],
  dashSelectedModelIds: [],
  dashRegisteredDatasets: [],
  dashRegisteredModels: [],
  dashDemoDatasets: null,
  dashPendingAction: null,
  _dashboardTrainMode: null,
  _dashboardAddDatasetMode: false,

  // ---- Autopilot -------------------------------------------------------
  _autopilotState: null,

  // ---- Image-viewer window handlers ------------------------------------
  _ivcWindowMoveHandler: null,
  _ivcWindowUpHandler: null,

  // ---- Inclusion -------------------------------------------------------
  inclusion: 0,
};

export default State;
