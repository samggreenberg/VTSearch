(function() {
  let medias = [];
  let votes = { good: [], bad: [], click_times: {}, learned_scores: {} };
  let goodVoteSet = new Set();
  let badVoteSet = new Set();
  function _rebuildVoteSets() {
    goodVoteSet = new Set(votes.good);
    badVoteSet = new Set(votes.bad);
  }
  let labelSortMode = "time-desc"; // default: newest first
  let selected = null;
  let sortOrder = null;   // null = default, or [{id, score}, ...]
  let sortMode = "text";  // "text" | "learned" | "load"
  let selectMode = "top"; // "top" | "hard"
  let threshold = null;    // threshold for Good/Bad boundary
  let sortTimer = null;
  let inclusion = 0;       // Inclusion setting: -10 to +10
  let loadedDetector = null; // Stores loaded detector model weights
  let datasetLoaded = false;
  let audioVolume = 1.0; // Persisted volume across media loads
  let volumeSaveTimer = null;
  let progressTimer = null;
  let progressEtaState = null; // { startTime, lastCurrent, lastTime } for ETA calculation
  let learnedSortController = null; // AbortController for in-flight background training
  let paragraphController = null;   // AbortController for in-flight paragraph content fetch
  let learnedSortDebounce = null;   // Debounce timer for background training
  let waveformAudioCtx = null;       // Shared AudioContext for waveform decoding
  let swipeAnimation = true;         // Swipe animation on vote (persisted setting)
  let isVoting = false;              // Re-entrance guard for castVote
  let _combineState = null;          // When non-null, we are in combine-datasets staging mode
  // Autopilot state machine: null when inactive, or {phase, goodToStart, badToStart, hardLabels, ...}
  // phase: "good" | "bad" | "hard" | "new" | "done"
  let _autopilotState = null;
  // Media type metadata fetched from /api/media-types at startup.
  // Keyed by type_id → { type_id, name, icon, tab_title, loops, ... }
  let mediaTypesMap = {};
  // Dashboard state
  let dashSelectedDataset = null;    // { name, label, ... } from demo list, or null
  let dashSelectedDetector = null;   // detector name string, or null
  // Registry-based multi-select state
  let dashSelectedDatasetIds = [];   // array of registry dataset IDs (for multi-select)
  let dashSelectedModelIds = [];     // array of registry model IDs (for multi-select)
  let dashRegisteredDatasets = [];   // cached from /api/datasets/registry
  let dashRegisteredModels = [];     // cached from /api/models/registry
  let dashDemoDatasets = null;       // cached demo dataset list from API
  let dashPendingAction = null;      // "label" | "detect" — set before loading a dataset
  let currentView = "welcome";       // "welcome" | "dashboard" | "labeling"
  // Dashboard train mode: null when inactive, or { model: detectorObj } for
  // the selected detector being trained in labeling mode
  let _dashboardTrainMode = null;
  // Tracks when the user is adding a dataset via the dashboard "+" button
  let _dashboardAddDatasetMode = false;
  // Local copy of favorite detectors list
  let favoriteDetectors = [];
  let autorunDetectors = [];
  const mediaList = document.getElementById("media-list");
  const center = document.getElementById("center");
  const goodList = document.getElementById("good-list");
  const badList = document.getElementById("bad-list");
  const textSortInput = document.getElementById("text-sort");
  const textSortWrap = document.getElementById("text-sort-wrap");
  const loadSortWrap = document.getElementById("load-sort-wrap");
  const loadSortDesc = document.getElementById("load-sort-desc");
  const learnedSortWrap = document.getElementById("learned-sort-wrap");
  const learnedSortDesc = document.getElementById("learned-sort-desc");
  // load-detector-file removed: Load Sort modal handles file picking now
  const learnedRadio = document.getElementById("learned-radio");
  const loadRadio = document.getElementById("load-radio");
  const sortStatus = document.getElementById("sort-status");
  const sortProgress = document.getElementById("sort-progress");
  const sortProgressFill = document.querySelector(".sort-progress-fill");
  let sortProgressTimer = null;
  let sortEtaState = null;

  // --- Accessibility: screen reader announcer ---
  const srAnnouncer = document.createElement("div");
  srAnnouncer.setAttribute("aria-live", "polite");
  srAnnouncer.setAttribute("aria-atomic", "true");
  srAnnouncer.className = "sr-only";
  srAnnouncer.id = "sr-announcer";
  document.body.appendChild(srAnnouncer);

  function announce(message) {
    // Clear then set to ensure screen readers pick up repeated messages
    srAnnouncer.textContent = "";
    setTimeout(() => { srAnnouncer.textContent = message; }, 100);
  }

  // --- Theme helper: read CSS custom property values for canvas drawing ---
  function themeColor(varName) {
    return getComputedStyle(document.documentElement).getPropertyValue(varName).trim();
  }

  // Dialog system delegated to static/dialogs.js (window.VTDialogs)
  const vtShowDialog = window.VTDialogs.vtShowDialog;
  const vtAlert = window.VTDialogs.vtAlert;
  const vtConfirm = window.VTDialogs.vtConfirm;
  const vtPrompt = window.VTDialogs.vtPrompt;

  function showSortProgress(label) {
    sortStatus.textContent = label;
    sortProgressFill.style.width = "";
    sortProgressFill.classList.remove("determinate");
    sortProgress.classList.add("active");
    sortEtaState = null;
  }

  function showSortProgressWithPolling(label) {
    showSortProgress(label);
    startSortProgressPolling();
  }

  function hideSortProgress() {
    stopSortProgressPolling();
    sortProgress.classList.remove("active");
    sortEtaState = null;
  }

  async function pollSortProgress() {
    try {
      const res = await fetch("/api/sort/progress");
      const progress = await res.json();
      if (progress.status === "idle") return;
      if (progress.total > 0) {
        const pct = Math.round((progress.current / progress.total) * 100);
        sortProgressFill.classList.add("determinate");
        sortProgressFill.style.width = `${pct}%`;

        // Calculate ETA for sort operations with many items
        const now = Date.now();
        if (!sortEtaState || sortEtaState.total !== progress.total) {
          sortEtaState = { startTime: now, startCurrent: progress.current, total: progress.total };
        }
        const elapsed = (now - sortEtaState.startTime) / 1000;
        const done = progress.current - sortEtaState.startCurrent;
        if (done > 0 && elapsed > 1 && progress.current < progress.total && progress.total >= 20) {
          const rate = done / elapsed;
          const remaining = (progress.total - progress.current) / rate;
          sortStatus.textContent = `${pct}% — ${formatETA(remaining)}`;
          return;
        }
      }
      if (progress.message) {
        sortStatus.textContent = progress.message;
      }
    } catch (_) {
      // ignore polling errors
    }
  }

  function startSortProgressPolling() {
    if (sortProgressTimer) return;
    sortProgressTimer = setInterval(pollSortProgress, 200);
  }

  function stopSortProgressPolling() {
    if (sortProgressTimer) {
      clearInterval(sortProgressTimer);
      sortProgressTimer = null;
    }
  }

  // ---- ETA formatting helper ----
  function formatETA(secondsRemaining) {
    if (secondsRemaining < 5) return "Less than 5 seconds remaining";
    if (secondsRemaining < 30) return "Less than 30 seconds remaining";
    if (secondsRemaining < 60) return "Less than a minute remaining";
    if (secondsRemaining < 90) return "About a minute remaining";
    if (secondsRemaining < 300) return `About ${Math.round(secondsRemaining / 60)} minutes remaining`;
    if (secondsRemaining < 3600) return `About ${Math.round(secondsRemaining / 60)} minutes remaining`;
    const hours = Math.floor(secondsRemaining / 3600);
    const mins = Math.round((secondsRemaining % 3600) / 60);
    if (hours === 1) return mins > 0 ? `About 1 hour ${mins} minutes remaining` : "About 1 hour remaining";
    return `About ${hours} hours remaining`;
  }

  const stripeOverview = document.getElementById("stripe-overview");
  const stripeContainer = document.getElementById("stripe-container");
  const inclusionSlider = document.getElementById("inclusion-slider");
  const inclusionValue = document.getElementById("inclusion-value");
  const calibrateCountInput = document.getElementById("calibrate-count-input");
  const calibrationFractionInput = document.getElementById("calibration-fraction-input");

  // Dataset management elements
  const datasetWelcome = document.getElementById("dataset-welcome");
  const datasetOptions = document.getElementById("dataset-options");
  const datasetProgress = document.getElementById("dataset-progress");
  const progressFill = document.getElementById("progress-fill");
  const progressText = document.getElementById("progress-text");
  const progressMessage = document.getElementById("progress-message");
  const progressEta = document.getElementById("progress-eta");
  const demoDatasetsDiv = document.getElementById("demo-datasets");
  const extendedImporterForm = document.getElementById("extended-importer-form");
  const backButton = document.getElementById("back-button");
  const loadFileBtn = document.getElementById("load-file-btn");
  const fileInput = document.getElementById("file-input");
  const datasetLoadColumn = document.getElementById("dataset-load-column");
  const datasetGenerateColumn = document.getElementById("dataset-generate-column");
  const datasetBar = document.getElementById("dataset-bar");
  const datasetInfo = document.getElementById("dataset-info");
  const leftPanel = document.getElementById("left-panel");
  const rightPanel = document.querySelector(".panel-right");
  const sortBar = document.getElementById("sort-bar");
  const trainDatasetBar = document.getElementById("train-dataset-bar");
  const trainDatasetName = document.getElementById("train-dataset-name");
  const trainDetectorBar = document.getElementById("train-detector-bar");
  const trainDetectorName = document.getElementById("train-detector-name");
  const trainExportDetectorBtn = document.getElementById("train-export-detector");
  const trainExportLabelsBtn = document.getElementById("train-export-labels");

  // Burger menu elements
  const burgerBtn = document.getElementById("burger-btn");
  const burgerDropdown = document.getElementById("burger-dropdown");
  const menuLabelsImport = document.getElementById("menu-labels-import");
  const menuLabelsStatus = document.getElementById("menu-labels-status");
  const menuDetectorImport = document.getElementById("menu-detector-import");
  const menuDetectorStatus = document.getElementById("menu-detector-status");
  const labelImporterModal = document.getElementById("label-importer-modal");
  const labelImporterModalClose = document.getElementById("label-importer-modal-close");
  const labelImporterList = document.getElementById("label-importer-list");
  const labelImporterFormDiv = document.getElementById("label-importer-form");
  const labelImporterBack = document.getElementById("label-importer-back");
  const labelExporterModal = document.getElementById("label-exporter-modal");
  const labelExporterModalClose = document.getElementById("label-exporter-modal-close");
  const labelExporterList = document.getElementById("label-exporter-list");
  const detectorExportModal = document.getElementById("detector-export-modal");
  const detectorExportModalClose = document.getElementById("detector-export-modal-close");
  const detectorExportList = document.getElementById("detector-export-list");
  const processorImporterModal = document.getElementById("processor-importer-modal");
  const processorImporterModalClose = document.getElementById("processor-importer-modal-close");
  const processorImporterList = document.getElementById("processor-importer-list");
  const processorImporterFormDiv = document.getElementById("processor-importer-form");
  const processorImporterBack = document.getElementById("processor-importer-back");
  const loadSortModal = document.getElementById("load-sort-modal");
  const loadSortModalClose = document.getElementById("load-sort-modal-close");
  const loadSortDetectorOptions = document.getElementById("load-sort-detector-options");
  const loadSortExampleOptions = document.getElementById("load-sort-example-options");
  const loadSortStatus = document.getElementById("load-sort-status");
  const loadSortDetectorFile = document.getElementById("load-sort-detector-file");
  const loadSortMediaFile = document.getElementById("load-sort-media-file");
  // autodetectModal, autodetectModalClose, autodetectSummary, autodetectResults,
  // copyResultsBtn — moved to static/results.js
  const autodetectProgressModal = document.getElementById("autodetect-progress-modal");
  const autodetectProgressText = document.getElementById("autodetect-progress-text");
  const autodetectProgressBar = document.getElementById("autodetect-progress-bar");

  // Examples editor modal elements
  const examplesEditorModal = document.getElementById("examples-editor-modal");
  const examplesEditorModalClose = document.getElementById("examples-editor-modal-close");
  const examplesEditorGrid = document.getElementById("examples-editor-grid");
  const examplesEditorType = document.getElementById("examples-editor-type");
  const examplesEditorAdd = document.getElementById("examples-editor-add");
  const examplesEditorSave = document.getElementById("examples-editor-save");
  const examplesEditorStatus = document.getElementById("examples-editor-status");
  const examplesMediaFile = document.getElementById("examples-media-file");

  // Autopilot examples elements
  const autopilotExamplesSection = document.getElementById("autopilot-examples-section");
  const autopilotExamplesEdit = document.getElementById("autopilot-examples-edit");
  const autopilotExamplesSummary = document.getElementById("autopilot-examples-summary");
  const autopilotStepsDiv = document.getElementById("autopilot-steps");

  // Settings modal elements
  const menuSettings = document.getElementById("menu-settings");
  const settingsModal = document.getElementById("settings-modal");
  const settingsModalClose = document.getElementById("settings-modal-close");
  const safeThresholdsCheckbox = document.getElementById("safe-thresholds-checkbox");
  const enrichDescCheckbox = document.getElementById("enrich-descriptions-checkbox");
  const settingsDefaultBtn = document.getElementById("settings-default-btn");
  const settingsImportBtn = document.getElementById("settings-import-btn");
  const settingsImportFile = document.getElementById("settings-import-file");
  const settingsExportBtn = document.getElementById("settings-export-btn");
  const swipeAnimationCheckbox = document.getElementById("swipe-animation-checkbox");
  const showThumbnailsLeftCheckbox = document.getElementById("show-thumbnails-left-checkbox");
  const showThumbnailsRightCheckbox = document.getElementById("show-thumbnails-right-checkbox");
  let showThumbnailsLeft = false;

  // Dashboard elements
  const dashboardView = document.getElementById("dashboard-view");
  const dashDatasetGrid = document.getElementById("dash-dataset-grid");
  const dashModelGrid = document.getElementById("dash-model-grid");
  const dashDatasetStatus = document.getElementById("dash-dataset-status");
  const dashLabelBtn = document.getElementById("dash-label-btn");
  const dashDetectBtn = document.getElementById("dash-detect-btn");
  const dashAddDatasetBtn = document.getElementById("dash-add-dataset-btn");

  const dashAddModelBtn = document.getElementById("dash-add-model-btn");
  const datasetImporterModal = document.getElementById("dataset-importer-modal");
  const datasetImporterModalClose = document.getElementById("dataset-importer-modal-close");
  const datasetImporterList = document.getElementById("dataset-importer-list");
  const datasetImporterFormDiv = document.getElementById("dataset-importer-form");
  const datasetImporterBack = document.getElementById("dataset-importer-back");
  const dashFileInput = document.getElementById("dash-file-input");
  // Dashboard progress elements — created dynamically as a table row
  let dashProgressFill = null;
  let dashProgressText = null;
  let dashProgressMessage = null;
  let dashProgressEta = null;
  const headerDashboardBtn = document.getElementById("header-dashboard-btn");
  const menuDashboard = document.getElementById("menu-dashboard");
  let showThumbnailsRight = true;
  const favMtCheckboxes = document.querySelectorAll("[data-media-type]");
  const autopilotTopGreensInput = document.getElementById("autopilot-top-greens-input");
  const autopilotHardRedsInput = document.getElementById("autopilot-hard-reds-input");

  // Left-panel tab switching
  const tabManual = document.getElementById("tab-manual");
  const tabAutopilot = document.getElementById("tab-autopilot");
  const tabPanelManual = document.getElementById("tab-panel-manual");
  const tabPanelAutopilot = document.getElementById("tab-panel-autopilot");

  if (tabManual && tabAutopilot) {
    tabManual.addEventListener("click", () => {
      tabManual.classList.add("active");
      tabManual.setAttribute("aria-selected", "true");
      tabAutopilot.classList.remove("active");
      tabAutopilot.setAttribute("aria-selected", "false");
      tabPanelManual.style.display = "";
      tabPanelAutopilot.style.display = "none";
      stopAutopilot();
    });
    tabAutopilot.addEventListener("click", () => {
      tabAutopilot.classList.add("active");
      tabAutopilot.setAttribute("aria-selected", "true");
      tabManual.classList.remove("active");
      tabManual.setAttribute("aria-selected", "false");
      tabPanelAutopilot.style.display = "";
      tabPanelManual.style.display = "none";
      refreshAutopilotExamples();
      startAutopilot();
    });
  }

  /**
   * Render a list of examples into containerEl.
   * Each example is {type, value}. Provides delete buttons; calls onChange(updatedArray) on mutation.
   */
  function renderExamplesGrid(containerEl, examples, onChange) {
    containerEl.innerHTML = "";
    if (!examples || examples.length === 0) {
      containerEl.innerHTML = '<div class="examples-empty">No examples yet.</div>';
      return;
    }
    examples.forEach((ex, i) => {
      const row = document.createElement("div");
      row.className = "example-row";
      const badge = document.createElement("span");
      badge.className = `example-type-badge type-${escapeHtml(ex.type || "text")}`;
      badge.textContent = ex.type || "text";
      const val = document.createElement("span");
      val.className = "example-value";
      val.title = ex.value || "";
      val.textContent = ex.value || "";
      const removeBtn = document.createElement("button");
      removeBtn.className = "example-remove";
      removeBtn.setAttribute("aria-label", "Remove example");
      removeBtn.textContent = "\u00D7";
      removeBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        const updated = examples.filter((_, j) => j !== i);
        onChange(updated);
      });
      row.appendChild(badge);
      row.appendChild(val);
      row.appendChild(removeBtn);
      containerEl.appendChild(row);
    });
  }

  /**
   * Prompt the user for a single example of the given type.
   * Returns {type, value} or null if cancelled.
   *
   * Standard types: "text", "media", "detector".
   * Importer types: "proc_imp:<name>" runs a processor importer,
   *                 "label_imp:<name>" runs a label importer (train a detector).
   * Importer types create a detector and return {type: "detector", value: name}.
   */
  async function promptForExample(type) {
    if (type === "text") {
      const val = await vtPrompt("Enter a text description for this example:", "");
      if (val && val.trim()) return { type: "text", value: val.trim() };
      return null;
    }
    if (type === "media") {
      // Fetch server media files and let user pick one
      let files = [];
      try {
        const res = await fetch("/api/server-media-files");
        if (res.ok) { const data = await res.json(); files = data.files || []; }
      } catch (_) { /* ignore */ }
      if (files.length === 0) {
        await vtAlert("No example media files found on server. Place files in data/example_media/ to use this option.", "warning");
        return null;
      }
      return new Promise((resolve) => {
        const overlay = document.createElement("div");
        overlay.className = "example-picker-overlay";
        overlay.innerHTML = `<div class="example-picker-panel">
          <div class="example-picker-header"><strong>Pick a server media file</strong></div>
          <div class="example-picker-list">${files.map((f, i) =>
            `<div class="load-sort-option option-card" data-idx="${i}" role="button" tabindex="0">
              <span class="option-card-icon">\uD83C\uDFB5</span>
              <div><div class="option-card-title">${escapeHtml(f.name)}</div>
              <div class="option-card-desc">${(f.size_bytes / 1024).toFixed(1)} KB</div></div>
            </div>`).join("")}</div>
          <button class="btn-sm" id="example-picker-cancel" style="margin-top:8px">Cancel</button>
        </div>`;
        document.body.appendChild(overlay);
        overlay.querySelectorAll("[data-idx]").forEach(el => {
          el.addEventListener("click", () => {
            const f = files[parseInt(el.dataset.idx, 10)];
            document.body.removeChild(overlay);
            resolve({ type: "media", value: f.filename });
          });
          el.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); el.click(); } });
        });
        overlay.querySelector("#example-picker-cancel").addEventListener("click", () => {
          document.body.removeChild(overlay);
          resolve(null);
        });
      });
    }
    if (type === "detector") {
      // Let user pick from existing autorun detectors
      const dets = autorunDetectors || [];
      if (dets.length === 0) {
        await vtAlert("No detectors found. Create a detector first.", "warning");
        return null;
      }
      return new Promise((resolve) => {
        const overlay = document.createElement("div");
        overlay.className = "example-picker-overlay";
        overlay.innerHTML = `<div class="example-picker-panel">
          <div class="example-picker-header"><strong>Pick a detector</strong></div>
          <div class="example-picker-list">${dets.map((d, i) =>
            `<div class="load-sort-option option-card" data-idx="${i}" role="button" tabindex="0">
              <span class="option-card-icon">\uD83E\uDD16</span>
              <div><div class="option-card-title">${escapeHtml(d.name)}</div>
              <div class="option-card-desc">${escapeHtml(d.media_type)}</div></div>
            </div>`).join("")}</div>
          <button class="btn-sm" id="example-picker-cancel" style="margin-top:8px">Cancel</button>
        </div>`;
        document.body.appendChild(overlay);
        overlay.querySelectorAll("[data-idx]").forEach(el => {
          el.addEventListener("click", () => {
            const d = dets[parseInt(el.dataset.idx, 10)];
            document.body.removeChild(overlay);
            resolve({ type: "detector", value: d.name });
          });
          el.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); el.click(); } });
        });
        overlay.querySelector("#example-picker-cancel").addEventListener("click", () => {
          document.body.removeChild(overlay);
          resolve(null);
        });
      });
    }
    // Processor importer: run the importer, create a detector, return as detector example
    if (type.startsWith("proc_imp:")) {
      const impName = type.slice("proc_imp:".length);
      return await promptForImporterExample("processor", impName);
    }
    // Label importer: run the label importer + train, create a detector, return as detector example
    if (type.startsWith("label_imp:")) {
      const impName = type.slice("label_imp:".length);
      return await promptForImporterExample("label", impName);
    }
    return null;
  }

  /**
   * Show a mini-form overlay for a processor or label importer, run the import,
   * and return {type: "detector", value: detectorName} on success, or null.
   *
   * @param {"processor"|"label"} kind
   * @param {string} importerName
   */
  async function promptForImporterExample(kind, importerName) {
    // Fetch the importer metadata
    let importer = null;
    try {
      const endpoint = kind === "label" ? "/api/label-importers" : "/api/processor-importers";
      const res = await fetch(endpoint);
      if (res.ok) {
        const list = await res.json();
        importer = list.find(i => i.name === importerName);
      }
    } catch (_) { /* ignore */ }
    if (!importer) {
      await vtAlert(`Importer "${importerName}" not found.`, "warning");
      return null;
    }

    return new Promise((resolve) => {
      const overlay = document.createElement("div");
      overlay.className = "example-picker-overlay";

      const title = kind === "label"
        ? `Import Labels: ${escapeHtml(importer.display_name)}`
        : escapeHtml(importer.display_name);
      const submitLabel = kind === "label" ? "Import & Train" : "Import";

      let fieldsHtml = `<div class="form-group">
        <label class="form-label">Detector Name *</label>
        <input type="text" name="name" placeholder="e.g. My Detector" class="form-input" required>
      </div>`;
      for (const field of importer.fields) {
        fieldsHtml += `<div class="form-group">`;
        fieldsHtml += `<label class="form-label">${escapeHtml(field.label)}${field.required ? " *" : ""}</label>`;
        if (field.field_type === "file") {
          fieldsHtml += `<input type="file" name="${escapeHtml(field.key)}" accept="${escapeHtml(field.accept)}" class="form-input" ${field.required ? "required" : ""}>`;
        } else if (field.field_type === "select") {
          fieldsHtml += `<select name="${escapeHtml(field.key)}" class="form-input">`;
          for (const opt of field.options) {
            fieldsHtml += `<option value="${escapeHtml(opt)}"${opt === field.default ? " selected" : ""}>${escapeHtml(opt || "(auto-detect)")}</option>`;
          }
          fieldsHtml += `</select>`;
        } else {
          const itype = field.field_type === "password" ? "password" : "text";
          const ph = escapeHtml(field.placeholder || field.description);
          fieldsHtml += `<input type="${itype}" name="${escapeHtml(field.key)}" value="${escapeHtml(field.default)}" placeholder="${ph}" class="form-input" ${field.required ? "required" : ""}>`;
        }
        if (field.description) fieldsHtml += `<div class="form-hint">${escapeHtml(field.description)}</div>`;
        fieldsHtml += `</div>`;
      }

      overlay.innerHTML = `<div class="example-picker-panel" style="max-width:420px">
        <div class="example-picker-header"><strong>${title}</strong></div>
        <form id="example-imp-form" style="padding:8px 0">
          ${fieldsHtml}
          <div id="example-imp-status" class="status-text compact"></div>
          <div style="display:flex;gap:8px;margin-top:8px">
            <button type="submit" class="btn-sm" style="flex:1">${submitLabel}</button>
            <button type="button" class="btn-sm" id="example-imp-cancel">Cancel</button>
          </div>
        </form>
      </div>`;
      document.body.appendChild(overlay);

      const statusEl = overlay.querySelector("#example-imp-status");

      overlay.querySelector("#example-imp-cancel").addEventListener("click", () => {
        document.body.removeChild(overlay);
        resolve(null);
      });

      overlay.querySelector("#example-imp-form").addEventListener("submit", async (e) => {
        e.preventDefault();
        const formEl = e.target;
        const detName = formEl.elements["name"].value.trim();
        if (!detName) { statusEl.textContent = "Name is required"; statusEl.style.color = "var(--color-bad)"; return; }

        statusEl.textContent = kind === "label" ? "Importing & training\u2026" : "Importing\u2026";
        statusEl.style.color = "var(--text-muted)";

        const hasFiles = importer.fields.some(f => f.field_type === "file");
        let body, headers = {};
        if (hasFiles) {
          body = new FormData(formEl);
        } else {
          const obj = { name: detName };
          for (const field of importer.fields) {
            obj[field.key] = formEl.elements[field.key].value;
          }
          body = JSON.stringify(obj);
          headers["Content-Type"] = "application/json";
        }

        const url = kind === "label"
          ? `/api/autorun-detectors/from-label-import/${encodeURIComponent(importer.name)}`
          : `/api/processor-importers/import/${encodeURIComponent(importer.name)}`;

        try {
          const res = await fetch(url, { method: "POST", headers, body });
          const result = await res.json();
          if (res.ok) {
            statusEl.textContent = `Created "${result.name || detName}"`;
            statusEl.style.color = "var(--color-good)";
            // Refresh detectors list so the new one is available
            try {
              const dRes = await fetch("/api/autorun-detectors");
              if (dRes.ok) { const d = await dRes.json(); favoriteDetectors = d.detectors || []; }
            } catch (_) { /* ignore */ }
            setTimeout(() => {
              document.body.removeChild(overlay);
              resolve({ type: "detector", value: result.name || detName });
            }, 600);
          } else {
            statusEl.textContent = result.error || "Import failed";
            statusEl.style.color = "var(--color-bad)";
          }
        } catch (err) {
          statusEl.textContent = `Error: ${err.message}`;
          statusEl.style.color = "var(--color-bad)";
        }
      });
    });
  }

  function refreshAutopilotExamples() {
    if (!_dashboardTrainMode || !_dashboardTrainMode.model) {
      if (autopilotExamplesSection) autopilotExamplesSection.style.display = "none";
      return;
    }
    const model = _dashboardTrainMode.model;
    if (autopilotExamplesSection) autopilotExamplesSection.style.display = "";
    if (autopilotExamplesSummary) {
      const examples = model.examples || [];
      if (examples.length === 0) {
        autopilotExamplesSummary.innerHTML = '<span class="ap-examples-empty">None yet</span>';
      } else {
        autopilotExamplesSummary.innerHTML = examples.map(ex =>
          `<span class="ap-example-chip"><span class="ap-chip-label">${escapeHtml(ex.value || ex.type || "example")}</span></span>`
        ).join("");
      }
    }
  }

  async function saveAutopilotExamples(model) {
    try {
      const url = model.trainable
        ? `/api/trainable-models/${encodeURIComponent(model.name)}/examples`
        : `/api/autorun-detectors/${encodeURIComponent(model.name)}/examples`;
      await fetch(url, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ examples: model.examples || [] }),
      });
    } catch (_) { /* ignore */ }
  }

  /**
   * Persist current labels to a trainable model's labelset (fire-and-forget).
   * Called after each vote in dashboard train mode.
   */
  function _persistTrainableModelLabels() {
    if (!_dashboardTrainMode || !_dashboardTrainMode.model) return;
    const model = _dashboardTrainMode.model;
    if (!model.trainable) return;
    fetch(`/api/trainable-models/${encodeURIComponent(model.name)}/labels`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
    }).catch(() => {}); // fire-and-forget
  }

  /**
   * Save trainable model labels (awaited version for explicit save points).
   */
  async function saveTrainableModelLabels() {
    if (!_dashboardTrainMode || !_dashboardTrainMode.model) return;
    const model = _dashboardTrainMode.model;
    if (!model.trainable) return;
    try {
      await fetch(`/api/trainable-models/${encodeURIComponent(model.name)}/labels`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      });
    } catch (_) { /* ignore */ }
  }

  // Stub functions for features that require the favorites modal HTML
  // (not yet present in index.html).  These prevent ReferenceErrors when
  // dashboard buttons or autopilot code call into the favorites UI.
  async function loadFavoriteDetectors() {
    try {
      const res = await fetch("/api/favorite-detectors");
      const data = await res.json();
      favoriteDetectors = data.detectors || [];
    } catch (_) {}
  }

  function loadFavImporterButtons() { /* no-op until favorites modal HTML is added */ }

  if (autopilotExamplesEdit) {
    autopilotExamplesEdit.addEventListener("click", () => {
      if (!_dashboardTrainMode || !_dashboardTrainMode.model) return;
      openExamplesEditorModal(_dashboardTrainMode.model);
    });
  }

  // ---- Autopilot state machine ----

  const _apSteps = [
    { id: "good", label: "Find Good labels from examples." },
    { id: "bad",  label: "Find Bad labels." },
    { id: "hard", label: "Label hard examples until smart." },
    { id: "new",  label: "Label new examples until diverse." },
  ];

  /**
   * Render the autopilot checklist into the steps div.
   * Shows all steps, highlights the active one with a detail sub-line,
   * crosses out completed steps, and dims future steps.
   */
  function renderAutopilotSteps() {
    if (!autopilotStepsDiv) return;
    if (!_autopilotState) {
      autopilotStepsDiv.innerHTML = "";
      return;
    }

    const st = _autopilotState;
    const phaseIdx = _apSteps.findIndex(s => s.id === st.phase);
    const isDone = st.phase === "done";

    let html = "";
    for (let i = 0; i < _apSteps.length; i++) {
      const step = _apSteps[i];
      let cls = "ap-step";
      if (isDone || i < phaseIdx) cls += " done";
      else if (i === phaseIdx) cls += " active";
      else cls += " future";

      html += `<div class="${cls}">${step.label}`;

      // Active step gets an expanded detail line
      if (!isDone && i === phaseIdx) {
        html += `<div class="ap-step-detail">${_autopilotDetail(st)}</div>`;
      }
      html += "</div>";
    }

    if (isDone) {
      html += `<div class="ap-step active" style="border-left-color:#2ecc71">Done! All indicators green.</div>`;
    }

    autopilotStepsDiv.innerHTML = html;
  }

  /**
   * Build the detail / progress string for the currently active phase.
   */
  function _autopilotDetail(st) {
    if (st.phase === "good") {
      const cur = votes.good.length;
      const target = st.goodToStart;
      const pct = Math.min(100, Math.round((cur / target) * 100));
      return `${cur}/${target} Good `
        + `<span class="ap-progress-bar"><span class="ap-progress-fill" style="width:${pct}%"></span></span>`;
    }
    if (st.phase === "bad") {
      const cur = votes.bad.length;
      const target = st.badToStart;
      const pct = Math.min(100, Math.round((cur / target) * 100));
      return `${cur}/${target} Bad `
        + `<span class="ap-progress-bar"><span class="ap-progress-fill" style="width:${pct}%"></span></span>`;
    }
    if (st.phase === "hard") {
      const n = st.hardLabels || 0;
      const smartSt = st.smartStatus || "";
      const stableSt = st.stableStatus || "";
      return `${n} hard labels applied `
        + `<span class="ap-indicator-dot" data-status="${smartSt}" title="Smart"></span>`
        + `<span class="ap-indicator-dot" data-status="${stableSt}" title="Stable"></span>`
        + ` Smart + Stable`;
    }
    if (st.phase === "new") {
      const frac = st.fracDiversity ?? 0;
      const pct = Math.min(100, Math.max(0, Math.round((frac / 4) * 100)));
      return `Diversity ${frac.toFixed(1)} / 4 `
        + `<span class="ap-progress-bar"><span class="ap-progress-fill" style="width:${pct}%"></span></span>`;
    }
    return "";
  }

  // Helper: set select-mode radio buttons and variable
  function _apSetSelectMode(mode) {
    selectMode = mode;
    document.querySelectorAll('input[name="select-mode"]').forEach(r => {
      r.checked = r.value === mode;
    });
  }

  // Helper: switch to learned sort mode
  function _apActivateLearnedSort() {
    sortMode = "learned";
    document.querySelectorAll('input[name="sort-mode"]').forEach(r => {
      r.checked = r.value === "learned";
    });
    textSortWrap.style.display = "none";
    learnedSortWrap.style.display = "";
    loadSortWrap.style.display = "none";
    updateLearnedSortDesc();
  }

  /**
   * Start the autopilot workflow.  Reads the first example from the current
   * model, triggers the appropriate sort, sets select-mode to Top, and enters
   * the "good" phase of the state machine.
   */
  async function startAutopilot() {
    if (!_dashboardTrainMode || !_dashboardTrainMode.model) {
      _autopilotState = null;
      renderAutopilotSteps();
      return;
    }
    const model = _dashboardTrainMode.model;
    const examples = model.examples || [];
    if (examples.length === 0) {
      _autopilotState = null;
      renderAutopilotSteps();
      return;
    }

    // Read settings for phase thresholds
    let goodToStart = 3;
    let badToStart = 4;
    try {
      const res = await fetch("/api/settings");
      if (res.ok) {
        const s = await res.json();
        if (typeof s.autopilot_top_greens === "number") goodToStart = s.autopilot_top_greens;
        if (typeof s.autopilot_hard_reds === "number") badToStart = s.autopilot_hard_reds;
      }
    } catch (_) { /* use defaults */ }

    _autopilotState = {
      phase: "good", goodToStart, badToStart,
      hardLabels: 0, smartStatus: "", stableStatus: "", spanStatus: "",
      fracDiversity: 0,
    };

    // Sort by the first example
    const firstExample = examples[0];

    // Set select mode to Top
    _apSetSelectMode("top");

    renderAutopilotSteps();

    if (firstExample.type === "text") {
      sortMode = "text";
      document.querySelectorAll('input[name="sort-mode"]').forEach(r => {
        r.checked = r.value === "text";
      });
      textSortWrap.style.display = "";
      learnedSortWrap.style.display = "none";
      loadSortWrap.style.display = "none";
      textSortInput.value = firstExample.value;
      await fetchTextSort(firstExample.value);
    } else if (firstExample.type === "media") {
      activateLoadSort("Example: " + firstExample.value);
      showSortProgress("Scoring with example media\u2026");
      try {
        const res = await fetch("/api/example-sort-server", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ filename: firstExample.value }),
        });
        if (!res.ok) throw new Error("Failed to sort by example");
        const data = await res.json();
        sortOrder = data.results.map(r => ({ id: r.id, score: r.similarity ?? r.score }));
        threshold = data.threshold;
        loadedDetector = { _example: true, _name: firstExample.value };
        hideSortProgress();
        sortStatus.textContent = `Threshold: ${(threshold * 100).toFixed(1)}%`;
        renderMediaList();
        const nextClip = findNextClip();
        if (nextClip) selectMedia(nextClip.id);
      } catch (err) {
        hideSortProgress();
        sortStatus.textContent = `Error: ${err.message}`;
      }
    } else if (firstExample.type === "detector") {
      try {
        const res = await fetch("/api/autorun-detectors");
        if (!res.ok) throw new Error("Failed to fetch detectors");
        const data = await res.json();
        const det = (data.detectors || []).find(d => d.name === firstExample.value);
        if (!det) throw new Error(`Detector "${firstExample.value}" not found`);
        loadedDetector = det;
        activateLoadSort("Detector: " + det.name);
        await fetchLoadedSort(true);
      } catch (err) {
        sortStatus.textContent = `Error: ${err.message}`;
      }
    }

    // Check if we already satisfy the first phase (e.g. from imported labels)
    checkAutopilotPhase();
  }

  /**
   * Check whether the autopilot should advance to the next phase based on
   * vote counts and indicator statuses.  Called after every vote and after
   * labeling-status updates.
   */
  function checkAutopilotPhase() {
    if (!_autopilotState) return;

    const st = _autopilotState;

    if (st.phase === "good") {
      if (votes.good.length >= st.goodToStart) {
        // Transition to Bad phase — switch select mode to Hard
        st.phase = "bad";
        _apSetSelectMode("hard");
        checkAutopilotPhase(); // re-check in case bad threshold also met
        return;
      }
    } else if (st.phase === "bad") {
      if (votes.bad.length >= st.badToStart) {
        // Transition to Hard phase — switch to learned sort + hard select
        st.phase = "hard";
        st.hardLabels = 0;
        _apActivateLearnedSort();
        _apSetSelectMode("hard");
        fetchLearnedSort(true);
        checkAutopilotPhase(); // re-check indicators
        return;
      }
    } else if (st.phase === "hard") {
      // Check if Smart AND Stable indicators are both green
      if (st.smartStatus === "green" && st.stableStatus === "green") {
        // Transition to New phase
        st.phase = "new";
        _apSetSelectMode("new");
        // Keep learned sort active; select mode = new uses diversity tree
        _fetchAndApplyDiversitySample();
        checkAutopilotPhase(); // re-check span
        return;
      }
    } else if (st.phase === "new") {
      // Check if Diverse (span) indicator is green
      if (st.spanStatus === "green") {
        st.phase = "done";
      }
    }

    renderAutopilotSteps();
  }

  /**
   * Called from applyLabelingStatus whenever indicator data arrives.
   * Feeds indicator statuses into the autopilot state machine so it can
   * decide when to transition from Hard→New and New→Done.
   */
  function _autopilotOnIndicatorUpdate(data) {
    if (!_autopilotState) return;
    const st = _autopilotState;
    if (data.smart) st.smartStatus = data.smart.status;
    if (data.stable) st.stableStatus = data.stable.status;
    if (data.span) {
      st.spanStatus = data.span.status;
      if (typeof data.span.fractional_level === "number") {
        st.fracDiversity = data.span.fractional_level;
      }
    }
    checkAutopilotPhase();
  }

  /**
   * Increment the hard-labels counter.  Called from castVote when
   * autopilot is in the "hard" phase.
   */
  function _autopilotCountHardLabel() {
    if (_autopilotState && _autopilotState.phase === "hard") {
      _autopilotState.hardLabels = (_autopilotState.hardLabels || 0) + 1;
    }
  }

  /** Fetch a diversity-tree sample and navigate to it. */
  async function _fetchAndApplyDiversitySample() {
    const data = await fetchDiversityTreeNext();
    if (data.id != null) {
      selectMedia(data.id);
    } else if (data.exhausted) {
      vtAlert("Diversity tree exhausted. All branches labeled.", "warning");
    }
  }

  function stopAutopilot() {
    _autopilotState = null;
    renderAutopilotSteps();
  }

  // ---- Trainable model label persistence ----

  /**
   * Fire-and-forget: persist current votes to the trainable model's labelset.
   * Called after every vote when in dashboard train mode.
   */
  function _persistTrainableModelLabels() {
    if (!_dashboardTrainMode || !_dashboardTrainMode.model) return;
    const name = _dashboardTrainMode.model.name;
    fetch(`/api/trainable-models/${encodeURIComponent(name)}/labels`, {
      method: "POST",
    }).catch(() => {});
  }

  /**
   * Save the current votes to the trainable model's labelset (awaitable).
   * Called when leaving train mode (e.g. navigating back to dashboard).
   */
  async function saveTrainableModelLabels() {
    if (!_dashboardTrainMode || !_dashboardTrainMode.model) return;
    const name = _dashboardTrainMode.model.name;
    try {
      await fetch(`/api/trainable-models/${encodeURIComponent(name)}/labels`, {
        method: "POST",
      });
    } catch (_) { /* ignore */ }
  }

  // ---- Dataset Management ----

  async function checkDatasetStatus() {
    const res = await fetch("/api/dataset/status");
    const status = await res.json();
    datasetLoaded = status.loaded;

    if (datasetLoaded) {
      const mtInfo = mediaTypesMap[status.media_type];
      const dupeSuffix = status.num_dupes ? ` (${status.num_dupes} dupes)` : "";
      datasetInfo.textContent = mtInfo
        ? `${mtInfo.icon} ${status.num_medias} ${mtInfo.name.toLowerCase()} loaded${dupeSuffix}`
        : `${status.num_medias} medias loaded${dupeSuffix}`;
      // Go to dashboard unless we're already in labeling view
      if (currentView !== "labeling") {
        showDashboard();
      }
    } else {
      showDashboard();
    }

    return status;
  }

  function showWelcomeScreen() {
    currentView = "welcome";
    // Clean up combine state if we're returning to the welcome screen.
    if (_combineState) {
      fetch("/api/dataset/staging", { method: "DELETE" }).catch(() => {});
      _combineState = null;
    }
    center.innerHTML = "";
    center.appendChild(datasetWelcome);
    datasetWelcome.classList.remove("wide");
    datasetWelcome.classList.remove("demo-mode");
    datasetWelcome.style.display = "flex";
    center.className = "panel-center";
    datasetOptions.style.display = "flex";
    datasetProgress.style.display = "none";
    demoDatasetsDiv.style.display = "none";
    extendedImporterForm.style.display = "none";
    backButton.style.display = "none";
    if (dashboardView) dashboardView.style.display = "none";
    const autodetectToggle = document.getElementById("autodetect-toggle");
    if (autodetectToggle) autodetectToggle.style.display = "";
    sortBar.style.display = "none";
    datasetBar.style.display = "none";
    trainDatasetBar.style.display = "none";
    trainDetectorBar.style.display = "none";
    mediaList.innerHTML = "";
    leftPanel.style.display = "none";
    if (rightPanel) rightPanel.style.display = "none";
    if (headerDashboardBtn) headerDashboardBtn.style.display = "none";
    stripeContainer.innerHTML = "";
    if (menuDashboard) menuDashboard.classList.remove("disabled");
    if (menuLabelsImport) menuLabelsImport.classList.add("disabled");
    if (menuDetectorImport) menuDetectorImport.classList.add("disabled");
  }

  function showMainUI() {
    currentView = "labeling";
    datasetWelcome.style.display = "none";
    if (dashboardView) dashboardView.style.display = "none";
    leftPanel.style.display = "";
    if (rightPanel) rightPanel.style.display = "";
    sortBar.style.display = "block";
    datasetBar.style.display = "flex";
    if (headerDashboardBtn) headerDashboardBtn.style.display = "";
    // Show train context bar when in dashboard train mode
    if (_dashboardTrainMode && _dashboardTrainMode.model) {
      if (trainDetectorBar) trainDetectorBar.style.display = "";
      if (trainDetectorName) trainDetectorName.textContent = _dashboardTrainMode.model.name;
    } else {
      if (trainDetectorBar) trainDetectorBar.style.display = "none";
    }
    if (!selected) {
      center.className = "panel-center empty";
      center.innerHTML = '<p>Select a media from the left panel</p>';
      announce("Dataset loaded. Select a media from the left panel to begin.");
    }
    if (menuDashboard) menuDashboard.classList.remove("disabled");
    if (menuLabelsImport) menuLabelsImport.classList.remove("disabled");
    if (menuDetectorImport) menuDetectorImport.classList.remove("disabled");
  }

  // ---- Dashboard view ----

  async function showDashboard() {
    currentView = "dashboard";
    _dashboardTrainMode = null;
    stopAutopilot();
    // Hide other views
    datasetWelcome.style.display = "none";
    leftPanel.style.display = "none";
    if (rightPanel) rightPanel.style.display = "none";
    sortBar.style.display = "none";
    datasetBar.style.display = "none";
    if (headerDashboardBtn) headerDashboardBtn.style.display = "none";
    // Import Labels is only for the labeling interface
    if (menuLabelsImport) menuLabelsImport.classList.add("disabled");
    if (menuDashboard) menuDashboard.classList.add("disabled");

    // Show dashboard
    center.innerHTML = "";
    center.appendChild(dashboardView);
    center.className = "panel-center";
    dashboardView.style.display = "flex";
    // Only hide the in-grid progress if no load is actively running
    if (!dashProgressTimer) hideDashGridProgress();

    // Update dataset status bar
    if (datasetLoaded) {
      const res = await fetch("/api/dataset/status");
      const status = await res.json();
      const mtInfo = mediaTypesMap[status.media_type];
      const dupeSuffix = status.num_dupes ? ` (${status.num_dupes} dupes)` : "";
      dashDatasetStatus.textContent = mtInfo
        ? `${mtInfo.icon} ${status.num_medias} ${mtInfo.name.toLowerCase()} loaded${dupeSuffix}`
        : `${status.num_medias} medias loaded${dupeSuffix}`;
      dashDatasetStatus.style.display = "";
    } else {
      dashDatasetStatus.style.display = "none";
    }

    // Populate grids
    await renderDashboardDatasets();
    await renderDashboardModels();
    updateDashboardButtons();
  }

  function updateDashboardButtons() {
    // --- Label button validation ---
    // Requires: exactly one dataset selected, exactly one model selected,
    // MediaTypes match, model is trainable
    if (dashLabelBtn) {
      const selDs = dashRegisteredDatasets.filter(d => dashSelectedDatasetIds.includes(d.id));
      const selMs = dashRegisteredModels.filter(m => dashSelectedModelIds.includes(m.id));
      let labelDisabled = false;
      let labelHint = "";

      if (selDs.length === 0 && !dashSelectedDataset) {
        labelDisabled = true;
        labelHint = "No dataset selected";
      } else if (selDs.length > 1) {
        labelDisabled = true;
        labelHint = "Select exactly one dataset";
      } else if (selMs.length === 0) {
        labelDisabled = true;
        labelHint = "No model selected";
      } else if (selMs.length > 1) {
        labelDisabled = true;
        labelHint = "Select exactly one model";
      } else if (selDs.length === 1 && selMs.length === 1) {
        const dsType = selDs[0].media_type;
        const mType = selMs[0].media_type;
        if (mType !== "any" && dsType !== mType) {
          labelDisabled = true;
          labelHint = "Dataset and Model types don't match";
        } else if (!selMs[0].trainable) {
          labelDisabled = true;
          labelHint = "Model is not trainable";
        }
      }

      dashLabelBtn.disabled = labelDisabled;
      // Show hint inside button
      const labelHintEl = dashLabelBtn.querySelector(".dash-btn-hint");
      if (labelHintEl) labelHintEl.textContent = labelHint;
    }

    // --- Find button validation ---
    // Requires: >= 1 dataset selected, >= 1 model selected, all same MediaType
    if (dashDetectBtn) {
      const selDs = dashRegisteredDatasets.filter(d => dashSelectedDatasetIds.includes(d.id));
      const selMs = dashRegisteredModels.filter(m => dashSelectedModelIds.includes(m.id));
      let findDisabled = false;
      let findHint = "";

      if (selDs.length === 0 && !dashSelectedDataset) {
        findDisabled = true;
        findHint = "Must select at least one dataset";
      } else if (selMs.length === 0) {
        findDisabled = true;
        findHint = "Must select at least one model";
      } else if (selDs.length >= 1 && selMs.length >= 1) {
        const allTypes = new Set([
          ...selDs.map(d => d.media_type),
          ...selMs.filter(m => m.media_type !== "any").map(m => m.media_type),
        ]);
        if (allTypes.size > 1) {
          findDisabled = true;
          findHint = "All selections must have the same media type";
        }
      }

      dashDetectBtn.disabled = findDisabled;
      // Show hint inside button
      const findHintEl = dashDetectBtn.querySelector(".dash-btn-hint");
      if (findHintEl) findHintEl.textContent = findHint;
    }
  }

  async function renderDashboardDatasets() {
    if (!dashDatasetGrid) return;

    // Fetch the full dataset registry
    try {
      const res = await fetch("/api/datasets/registry");
      const data = await res.json();
      dashRegisteredDatasets = data.datasets || [];
    } catch (_) {
      dashRegisteredDatasets = [];
    }

    // Auto-select when exactly one dataset exists
    if (dashRegisteredDatasets.length === 1 && dashSelectedDatasetIds.length === 0) {
      dashSelectedDatasetIds = [dashRegisteredDatasets[0].id];
    }

    if (dashRegisteredDatasets.length === 0) {
      dashDatasetGrid.innerHTML = '<p style="color:var(--text-muted); padding:16px;">No datasets yet. Use "+" to load one.</p>';
      return;
    }

    const mediaIcons = Object.fromEntries(Object.entries(mediaTypesMap).map(([k, v]) => [k, v.icon]));

    dashDatasetGrid.innerHTML = `<table class="dash-dataset-table">
      <thead><tr>
        <th>Name</th>
        <th>Type</th>
        <th>Items</th>
        <th>Loaded</th>
        <th class="col-actions-header"></th>
      </tr></thead>
      <tbody></tbody></table>`;

    const tbody = dashDatasetGrid.querySelector("tbody");

    dashRegisteredDatasets.forEach(ds => {
      const icon = mediaIcons[ds.media_type] || "";
      const typeName = mediaTypesMap[ds.media_type] ? mediaTypesMap[ds.media_type].name : ds.media_type || "media";
      const isSelected = dashSelectedDatasetIds.includes(ds.id);
      const isLoaded = !!ds.loaded;
      const tr = document.createElement("tr");
      tr.className = "dash-dataset-row" + (isSelected ? " dash-selected" : "");
      tr.setAttribute("role", "button");
      tr.setAttribute("tabindex", "0");

      const nameTd = document.createElement("td");
      nameTd.className = "col-name";
      nameTd.innerHTML = `<span class="dash-name-text">${escapeHtml(ds.name)}</span><button class="btn-icon dash-rename-btn" title="Rename" aria-label="Rename dataset">&#9998;</button>`;
      tr.appendChild(nameTd);
      tr.insertAdjacentHTML("beforeend", `
        <td class="col-type">${escapeHtml(icon)} ${escapeHtml(typeName)}</td>
        <td class="col-count">${ds.num_items || 0}</td>
        <td class="col-loaded">${isLoaded ? '<span style="color:var(--color-good)">✓</span>' : ''}</td>
        <td class="col-actions"><button class="btn-icon btn-icon-danger dash-delete-btn" title="Remove dataset" aria-label="Remove dataset">&#128465;</button></td>
      `);

      // Inline rename
      const renameBtn = tr.querySelector(".dash-rename-btn");
      renameBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        const nameSpan = nameTd.querySelector(".dash-name-text");
        const current = nameSpan.textContent;
        nameSpan.style.display = "none";
        renameBtn.style.display = "none";
        const input = document.createElement("input");
        input.type = "text";
        input.className = "dash-rename-input";
        input.value = current;
        nameTd.insertBefore(input, nameSpan);
        input.focus();
        input.select();
        const commit = async () => {
          const newName = input.value.trim();
          if (newName && newName !== current) {
            try {
              await fetch(`/api/datasets/registry/${encodeURIComponent(ds.id)}/rename`, {
                method: "PUT",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ name: newName }),
              });
              ds.name = newName;
              nameSpan.textContent = newName;
            } catch (_) {}
          }
          input.remove();
          nameSpan.style.display = "";
          renameBtn.style.display = "";
        };
        input.addEventListener("blur", commit);
        input.addEventListener("keydown", (ev) => {
          if (ev.key === "Enter") { ev.preventDefault(); input.blur(); }
          if (ev.key === "Escape") { input.value = current; input.blur(); }
        });
      });

      // Delete dataset
      const deleteBtn = tr.querySelector(".dash-delete-btn");
      deleteBtn.addEventListener("click", async (e) => {
        e.stopPropagation();
        if (!(await vtConfirm("Remove this dataset? Its saved file will be deleted."))) return;
        try {
          await fetch(`/api/datasets/registry/${encodeURIComponent(ds.id)}`, { method: "DELETE" });
          dashSelectedDatasetIds = dashSelectedDatasetIds.filter(id => id !== ds.id);
          if (ds.loaded) datasetLoaded = false;
          await renderDashboardDatasets();
          updateDashboardButtons();
        } catch (_) {}
      });

      // Multi-select click (toggle)
      tr.addEventListener("click", (e) => {
        if (e.ctrlKey || e.metaKey) {
          // Toggle individual selection
          if (dashSelectedDatasetIds.includes(ds.id)) {
            dashSelectedDatasetIds = dashSelectedDatasetIds.filter(id => id !== ds.id);
          } else {
            dashSelectedDatasetIds.push(ds.id);
          }
        } else {
          // Single-click replaces selection
          if (dashSelectedDatasetIds.length === 1 && dashSelectedDatasetIds[0] === ds.id) {
            dashSelectedDatasetIds = [];
          } else {
            dashSelectedDatasetIds = [ds.id];
          }
        }
        renderDashboardDatasets();
        updateDashboardButtons();
      });
      tr.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") { e.preventDefault(); tr.click(); }
      });

      tbody.appendChild(tr);
    });
  }

  async function renderDashboardModels() {
    if (!dashModelGrid) return;

    // Fetch the full model registry
    try {
      const res = await fetch("/api/models/registry");
      const data = await res.json();
      dashRegisteredModels = data.models || [];
    } catch (_) {
      dashRegisteredModels = [];
    }

    // Also fetch autorun detectors for backward compat
    try {
      const res = await fetch("/api/autorun-detectors");
      const data = await res.json();
      autorunDetectors = data.detectors || [];
    } catch (_) {}

    // Auto-select when exactly one model exists
    if (dashRegisteredModels.length === 1 && dashSelectedModelIds.length === 0) {
      dashSelectedModelIds = [dashRegisteredModels[0].id];
    }

    if (dashRegisteredModels.length === 0) {
      dashModelGrid.innerHTML = '<p style="color:var(--text-muted); padding:16px;">No models yet. Use "+" to create one.</p>';
      return;
    }

    const mediaIcons = Object.fromEntries(Object.entries(mediaTypesMap).map(([k, v]) => [k, v.icon]));

    dashModelGrid.innerHTML = `<table class="dash-model-table">
      <thead><tr>
        <th data-sort="name">Name<span class="sort-arrow"></span></th>
        <th data-sort="media_type">Type<span class="sort-arrow"></span></th>
        <th data-sort="num_training" style="text-align:right"># Training<span class="sort-arrow"></span></th>
        <th>Loaded</th>
        <th class="col-actions-header"></th>
      </tr></thead><tbody></tbody></table>`;

    const table = dashModelGrid.querySelector(".dash-model-table");
    let modelSort = { key: "name", asc: true };

    function renderModelRows() {
      const sorted = [...dashRegisteredModels].sort((a, b) => {
        let va = a[modelSort.key], vb = b[modelSort.key];
        if (modelSort.key === "num_training" || modelSort.key === "created_at") {
          return modelSort.asc ? ((va || 0) - (vb || 0)) : ((vb || 0) - (va || 0));
        }
        va = String(va || "").toLowerCase(); vb = String(vb || "").toLowerCase();
        return modelSort.asc ? va.localeCompare(vb) : vb.localeCompare(va);
      });

      const tbody = table.querySelector("tbody");
      tbody.innerHTML = "";
      sorted.forEach(m => {
        const icon = mediaIcons[m.media_type] || "";
        const isSelected = dashSelectedModelIds.includes(m.id);
        const isLoaded = !!m.loaded;
        const trainingText = m.trainable ? String(m.num_training || 0) : "-";
        const tr = document.createElement("tr");
        tr.className = "dash-model-row" + (isSelected ? " dash-selected" : "");
        tr.setAttribute("role", "button");
        tr.setAttribute("tabindex", "0");

        const nameTd = document.createElement("td");
        nameTd.className = "col-name";
        nameTd.innerHTML = `<span class="dash-name-text">${escapeHtml(m.name)}</span><button class="btn-icon dash-rename-btn" title="Rename" aria-label="Rename model">&#9998;</button>`;
        tr.appendChild(nameTd);
        tr.insertAdjacentHTML("beforeend", `
          <td class="col-type">${escapeHtml(icon)} ${escapeHtml(m.media_type)}</td>
          <td class="col-num-training" style="text-align:right">${escapeHtml(trainingText)}</td>
          <td class="col-loaded">${isLoaded ? '<span style="color:var(--color-good)">✓</span>' : ''}</td>
          <td class="col-actions"><button class="btn-icon btn-icon-danger dash-delete-btn" title="Remove model" aria-label="Remove model">&#128465;</button></td>
        `);

        // Inline rename
        const renameBtn = tr.querySelector(".dash-rename-btn");
        renameBtn.addEventListener("click", (e) => {
          e.stopPropagation();
          const nameSpan = nameTd.querySelector(".dash-name-text");
          const current = nameSpan.textContent;
          nameSpan.style.display = "none";
          renameBtn.style.display = "none";
          const input = document.createElement("input");
          input.type = "text";
          input.className = "dash-rename-input";
          input.value = current;
          nameTd.insertBefore(input, nameSpan);
          input.focus();
          input.select();
          const commit = async () => {
            const newName = input.value.trim();
            if (newName && newName !== current) {
              try {
                await fetch(`/api/models/registry/${encodeURIComponent(m.id)}/rename`, {
                  method: "PUT",
                  headers: { "Content-Type": "application/json" },
                  body: JSON.stringify({ name: newName }),
                });
                m.name = newName;
                nameSpan.textContent = newName;
              } catch (_) {}
            }
            input.remove();
            nameSpan.style.display = "";
            renameBtn.style.display = "";
          };
          input.addEventListener("blur", commit);
          input.addEventListener("keydown", (ev) => {
            if (ev.key === "Enter") { ev.preventDefault(); input.blur(); }
            if (ev.key === "Escape") { input.value = current; input.blur(); }
          });
        });

        // Delete model
        const deleteBtn = tr.querySelector(".dash-delete-btn");
        deleteBtn.addEventListener("click", async (e) => {
          e.stopPropagation();
          if (!(await vtConfirm(`Delete model "${m.name}"? This cannot be undone.`))) return;
          try {
            await fetch(`/api/models/registry/${encodeURIComponent(m.id)}`, { method: "DELETE" });
            dashSelectedModelIds = dashSelectedModelIds.filter(id => id !== m.id);
            await renderDashboardModels();
            updateDashboardButtons();
          } catch (_) {}
        });

        // Multi-select click (toggle)
        tr.addEventListener("click", (e) => {
          if (e.ctrlKey || e.metaKey) {
            if (dashSelectedModelIds.includes(m.id)) {
              dashSelectedModelIds = dashSelectedModelIds.filter(id => id !== m.id);
            } else {
              dashSelectedModelIds.push(m.id);
            }
          } else {
            if (dashSelectedModelIds.length === 1 && dashSelectedModelIds[0] === m.id) {
              dashSelectedModelIds = [];
            } else {
              dashSelectedModelIds = [m.id];
            }
          }
          renderModelRows();
          updateDashboardButtons();
        });
        tr.addEventListener("keydown", (e) => {
          if (e.key === "Enter" || e.key === " ") { e.preventDefault(); tr.click(); }
        });

        tbody.appendChild(tr);
      });

      // Sort arrows
      table.querySelectorAll("th[data-sort]").forEach(th => {
        const arrow = th.querySelector(".sort-arrow");
        if (th.dataset.sort === modelSort.key) {
          arrow.textContent = modelSort.asc ? " \u25B2" : " \u25BC";
        } else {
          arrow.textContent = "";
        }
      });
    }

    table.querySelectorAll("th[data-sort]").forEach(th => {
      th.addEventListener("click", () => {
        const key = th.dataset.sort;
        if (modelSort.key === key) {
          modelSort.asc = !modelSort.asc;
        } else {
          modelSort = { key, asc: true };
        }
        renderModelRows();
      });
    });

    renderModelRows();
    updateDashboardButtons();
  }

  // Show a progress row inside the dataset table (keeps existing rows visible)
  function showDashGridProgress(message) {
    // Remove any existing progress row first
    hideDashGridProgress();
    // Ensure the table exists; if the grid is empty, create a minimal table
    let tbody = dashDatasetGrid.querySelector("tbody");
    if (!tbody) {
      dashDatasetGrid.innerHTML = `<table class="dash-dataset-table">
        <thead><tr>
          <th>Name</th><th>Type</th><th>Items</th><th>Loaded</th><th class="col-actions-header"></th>
        </tr></thead><tbody></tbody></table>`;
      tbody = dashDatasetGrid.querySelector("tbody");
    }
    const tr = document.createElement("tr");
    tr.id = "dash-progress-row";
    tr.className = "dash-progress-row";
    tr.innerHTML = `<td colspan="5" class="dash-progress-cell" role="status" aria-live="polite">
      <div class="progress-bar" role="progressbar" aria-valuenow="0" aria-valuemin="0" aria-valuemax="100">
        <div class="progress-fill indeterminate" id="dash-progress-fill" style="width:0%"></div>
        <div class="progress-text" id="dash-progress-text" aria-hidden="true"></div>
      </div>
      <div class="progress-message" id="dash-progress-message">${escapeHtml(message || "Loading...")}</div>
      <div class="progress-eta" id="dash-progress-eta"></div>
    </td>`;
    tbody.appendChild(tr);
    // Update element references
    dashProgressFill = tr.querySelector("#dash-progress-fill");
    dashProgressText = tr.querySelector("#dash-progress-text");
    dashProgressMessage = tr.querySelector("#dash-progress-message");
    dashProgressEta = tr.querySelector("#dash-progress-eta");
    dashProgressMessage.style.color = "var(--text-secondary)";
  }

  // Remove the progress row from the dataset table
  function hideDashGridProgress() {
    const row = document.getElementById("dash-progress-row");
    if (row) row.remove();
    dashProgressFill = null;
    dashProgressText = null;
    dashProgressMessage = null;
    dashProgressEta = null;
  }

  // Dashboard: load a dataset from the selected demo, then run a callback
  async function dashLoadSelectedDataset(callback) {
    if (!dashSelectedDataset) return;

    showDashGridProgress("Loading...");

    try {
      const res = await fetch("/api/dataset/load-demo", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: dashSelectedDataset.name }),
      });
      if (!res.ok) {
        const error = await res.json();
        if (dashProgressMessage) { dashProgressMessage.textContent = `Error: ${error.error}`; dashProgressMessage.style.color = "var(--color-bad)"; }
        return;
      }
    } catch (e) {
      if (dashProgressMessage) { dashProgressMessage.textContent = `Error: ${e.message}`; dashProgressMessage.style.color = "var(--color-bad)"; }
      return;
    }

    // Poll progress until done
    dashPendingAction = callback;
    startDashProgressPolling();
  }

  let dashProgressTimer = null;

  function startDashProgressPolling() {
    if (dashProgressTimer) clearInterval(dashProgressTimer);
    dashProgressTimer = setInterval(pollDashProgress, 500);
  }

  function stopDashProgressPolling() {
    if (dashProgressTimer) { clearInterval(dashProgressTimer); dashProgressTimer = null; }
  }

  async function pollDashProgress() {
    try {
      const res = await fetch("/api/dataset/progress");
      const progress = await res.json();

      if (progress.error) {
        stopDashProgressPolling();
        if (dashProgressMessage) { dashProgressMessage.textContent = `Error: ${progress.error}`; dashProgressMessage.style.color = "var(--color-bad)"; }
        if (dashProgressFill) dashProgressFill.classList.remove("indeterminate");
        return;
      }

      if (progress.pct != null && dashProgressFill) {
        dashProgressFill.classList.remove("indeterminate");
        dashProgressFill.style.width = `${progress.pct}%`;
        if (dashProgressText) dashProgressText.textContent = `${progress.pct}%`;
      }
      if (progress.message && dashProgressMessage) {
        dashProgressMessage.textContent = progress.message;
        dashProgressMessage.style.color = "var(--text-secondary)";
      }

      if (progress.status === "idle") {
        stopDashProgressPolling();
        hideDashGridProgress();

        // Refresh dataset state
        await checkDatasetStatus();
        if (datasetLoaded) {
          await renderDashboardDatasets();
          await fetchMedias();
          await fetchVotes();

          const cb = dashPendingAction;
          dashPendingAction = null;
          if (typeof cb === "function") await cb();
        }
      }
    } catch (_) {}
  }

  function showProgress() {
    datasetOptions.style.display = "none";
    demoDatasetsDiv.style.display = "none";
    extendedImporterForm.style.display = "none";
    datasetProgress.style.display = "block";
    backButton.style.display = "none";
    // Reset progress bar to indeterminate state
    progressFill.style.width = "0%";
    progressFill.classList.add("indeterminate");
    progressText.textContent = "";
    progressMessage.textContent = "Loading...";
    progressMessage.style.color = "var(--text-secondary)";
    progressEta.textContent = "";
    progressEtaState = null;
  }

  async function pollProgress() {
    let res, progress;
    try {
      res = await fetch("/api/dataset/progress");
      progress = await res.json();
    } catch (_) {
      return; // Network error — retry on next poll interval
    }

    if (progress.error) {
      stopProgressPolling();
      if (_combineState) {
        showCombineDatasetsForm();
        vtAlert(progress.error, "warning");
      } else {
        showWelcomeScreen();
        vtAlert(progress.error, "warning");
      }
      return;
    }

    if (progress.status === "idle") {
      stopProgressPolling();

      // If we are in combine-datasets staging mode, handle the staging result
      // instead of loading the dataset into the training UI.
      if (_combineState && progress.staging_result) {
        _combineState.push(progress.staging_result);
        showCombineDatasetsForm();
        return;
      }
      if (_combineState && !progress.staging_result) {
        // Staging failed or produced no result – return to the combine form.
        showCombineDatasetsForm();
        return;
      }

      await checkDatasetStatus();
      if (datasetLoaded) {
        // Dashboard add-dataset mode: capture info and return to dashboard
        if (_dashboardAddDatasetMode) {
          _dashboardAddDatasetMode = false;
          // Dataset is auto-registered by the backend on load;
          // clear from active memory so the dashboard can manage it.
          await fetch("/api/dataset/clear", { method: "POST" });
          medias = [];
          votes = { good: [], bad: [], click_times: {}, learned_scores: {} };
          selected = null;
          datasetLoaded = false;
          showDashboard();
          return;
        }

        // Dashboard train mode: dataset loaded, now apply labels and enter labeling UI
        if (_dashboardTrainMode) {
          const trainInfo = _dashboardTrainMode;
          await fetchMedias();

          // Clear any leftover votes so only this model's labels are active
          try { await fetch("/api/votes/clear", { method: "POST" }); } catch (_) {}

          // Import labels from the trainable model's labelset
          try {
            const modelRes = await fetch(`/api/trainable-models/${encodeURIComponent(trainInfo.model.name)}`);
            if (modelRes.ok) {
              const modelData = await modelRes.json();
              const labels = modelData.labelset && modelData.labelset.labels;
              if (labels && labels.length > 0) {
                await fetch("/api/labels/import", {
                  method: "POST",
                  headers: { "Content-Type": "application/json" },
                  body: JSON.stringify({ labels }),
                });
              }
            }
          } catch (_) { /* ignore label import errors */ }

          await fetchVotes();

          // Set text sort mode and trigger sort with the model's text query or first text example
          const exQuery = trainInfo.model.text_query
            || (trainInfo.model.examples && trainInfo.model.examples.length > 0 && trainInfo.model.examples[0].type === "text" ? trainInfo.model.examples[0].value : "");
          sortMode = "text";
          textSortWrap.style.display = "";
          learnedSortWrap.style.display = "none";
          loadSortWrap.style.display = "none";
          // Update radio buttons to reflect text sort mode
          document.querySelectorAll('input[name="sort-mode"]').forEach(r => {
            r.checked = r.value === "text";
          });
          textSortInput.value = exQuery;
          if (exQuery) {
            fetchTextSort(exQuery);
          }

          showMainUI();
          if (tabAutopilot) {
            tabAutopilot.click();
          }
          // Select first media
          if (medias.length > 0 && !selected) {
            selectMedia(medias[0].id);
          }
          return;
        }

        await fetchMedias();
        await fetchVotes();
        // Dataset loaded — dashboard is now shown via checkDatasetStatus.
        // Auto-select is deferred until user clicks "Label".
      }
      return;
    }

    // Update progress bar
    if (progress.total > 0) {
      const percentage = Math.round((progress.current / progress.total) * 100);
      progressFill.classList.remove("indeterminate");
      progressFill.style.width = `${percentage}%`;
      progressText.textContent = `${percentage}%`;

      // Only calculate ETA during the "embedding" phase.  Earlier phases like
      // warmup also report a known total (e.g. 1/3, 2/3, 3/3) to drive their
      // own determinate bar, but their per-step timing must not seed the ETA
      // for the subsequent, much longer embedding loop.
      const now = Date.now();
      if (progress.status === "embedding") {
        if (!progressEtaState || progressEtaState.total !== progress.total) {
          progressEtaState = { startTime: now, startCurrent: progress.current, total: progress.total };
        }
        const elapsed = (now - progressEtaState.startTime) / 1000;
        const done = progress.current - progressEtaState.startCurrent;
        if (done > 0 && elapsed > 1 && progress.current < progress.total) {
          const rate = done / elapsed;
          const remaining = (progress.total - progress.current) / rate;
          progressEta.textContent = formatETA(remaining);
        } else if (progress.current >= progress.total) {
          progressEta.textContent = "";
        }
      } else {
        // Non-embedding phase with a known total (e.g. warmup): show the
        // determinate bar but suppress ETA and clear any stale ETA state so
        // embedding always starts with a fresh timer.
        progressEtaState = null;
        progressEta.textContent = "";
      }
    } else {
      // Indeterminate state - no total known yet
      progressFill.classList.add("indeterminate");
      progressText.textContent = "";
      progressEta.textContent = "";
    }
    progressMessage.textContent = progress.message || "Loading...";
    progressMessage.style.color = "var(--text-secondary)";
  }

  function startProgressPolling() {
    if (progressTimer) return;
    showProgress();
    progressTimer = setInterval(pollProgress, 500);
  }

  function stopProgressPolling() {
    if (progressTimer) {
      clearInterval(progressTimer);
      progressTimer = null;
    }
    progressFill.classList.remove("indeterminate");
    progressEta.textContent = "";
    progressEtaState = null;
  }

  // Load from file
  loadFileBtn.addEventListener("click", () => {
    fileInput.click();
  });

  fileInput.addEventListener("change", async () => {
    const file = fileInput.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    startProgressPolling();

    try {
      const res = await fetch("/api/dataset/load-file", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const error = await res.json();
        progressMessage.textContent = `Error: ${error.error}`;
        progressMessage.style.color = "var(--color-bad)";
        stopProgressPolling();
      }
    } catch (e) {
      progressMessage.textContent = `Error: ${e.message}`;
      progressMessage.style.color = "var(--color-bad)";
      stopProgressPolling();
    }

    fileInput.value = "";
  });

  async function loadDemo(name) {
    startProgressPolling();

    try {
      const res = await fetch("/api/dataset/load-demo", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name }),
      });

      if (!res.ok) {
        const error = await res.json();
        progressMessage.textContent = `Error: ${error.error}`;
        progressMessage.style.color = "var(--color-bad)";
        stopProgressPolling();
      }
    } catch (e) {
      progressMessage.textContent = `Error: ${e.message}`;
      progressMessage.style.color = "var(--color-bad)";
      stopProgressPolling();
    }
  }

  // ---- Extended importers (auto-discovered via /api/dataset/importers) ----

  async function loadExtendedImporters() {
    try {
      const res = await fetch("/api/dataset/importers");
      if (!res.ok) return;
      const data = await res.json();
      for (const importer of data.importers) {
        const btn = document.createElement("button");
        btn.className = "dataset-option";
        btn.innerHTML = `<h3>${escapeHtml(importer.icon || "🔌")} ${escapeHtml(importer.display_name)}</h3><p>${escapeHtml(importer.description)}</p>`;
        btn.addEventListener("click", () => showExtendedImporterForm(importer));
        datasetGenerateColumn.appendChild(btn);
      }
    } catch (_) {
      // Extended importers are optional – silently ignore failures.
    }

    // Append the Combine Existing Datasets button
    const combineBtnEl = document.createElement("button");
    combineBtnEl.className = "dataset-option";
    combineBtnEl.id = "combine-datasets-btn";
    combineBtnEl.innerHTML = `<h3>\uD83D\uDD00 Combine Existing Datasets</h3><p>Merge multiple .pkl datasets into one, skipping duplicates</p>`;
    combineBtnEl.addEventListener("click", () => showCombineDatasetsForm());
    datasetLoadColumn.appendChild(combineBtnEl);

    // Append the Load Demo Dataset button after all dynamic importers
    const demoBtnEl = document.createElement("button");
    demoBtnEl.className = "dataset-option";
    demoBtnEl.id = "load-demo-btn";
    demoBtnEl.innerHTML = `<h3>🏆 Load Demo Dataset</h3><p>Choose from a selection of pre-configured demo datasets</p>`;
    demoBtnEl.addEventListener("click", async () => {
      datasetOptions.style.display = "none";
      demoDatasetsDiv.style.display = "flex";
      backButton.style.display = "block";
      datasetWelcome.classList.add("wide");
      datasetWelcome.classList.add("demo-mode");

      // Fetch demo datasets
      try {
        const res = await fetch("/api/dataset/demo-list");
        if (!res.ok) {
          throw new Error(`Server returned ${res.status}`);
        }
        const data = await res.json();

        demoDatasetsDiv.innerHTML = "";

        // Group datasets by media type and display as sortable tables.
        // Media type metadata comes from the registry-driven mediaTypesMap.
        const grouped = {};
        data.datasets.forEach(ds => {
          const mt = ds.media_type || "audio";
          if (!grouped[mt]) grouped[mt] = [];
          grouped[mt].push(ds);
        });
        // Use the registration order from the registry (fetched at startup)
        const mediaOrder = Object.keys(mediaTypesMap).filter(mt => (grouped[mt] || []).length > 0);

        // Status sort order: ready=0, needs_embedding=1, needs_download=2
        const statusOrder = { ready: 0, needs_embedding: 1, needs_download: 2 };

        const sortColumns = [
          { key: "label",          label: "Name" },
          { key: "num_files",      label: "# Media" },
          { key: "num_categories", label: "# Cat." },
          { key: "description",    label: "Description" },
          { key: "status",         label: "Readiness" },
        ];

        function buildStatusBadge(st) {
          if (st === "ready") return '<span class="ready-badge">Ready</span>';
          if (st === "needs_embedding") return '<span class="embedding-badge">Needs Embed</span>';
          return '<span class="download-badge">Needs Download</span>';
        }

        function renderTable(items, section) {
          const sortState = section._demoSort || { key: "label", asc: true };
          const sorted = [...items].sort((a, b) => {
            let va = a[sortState.key], vb = b[sortState.key];
            if (sortState.key === "status") { va = statusOrder[va] ?? 3; vb = statusOrder[vb] ?? 3; }
            if (typeof va === "number" && typeof vb === "number") return sortState.asc ? va - vb : vb - va;
            va = String(va || "").toLowerCase(); vb = String(vb || "").toLowerCase();
            return sortState.asc ? va.localeCompare(vb) : vb.localeCompare(va);
          });

          const tbody = section.querySelector("tbody");
          tbody.innerHTML = "";
          sorted.forEach(ds => {
            const st = ds.status || (ds.ready ? "ready" : "needs_download");
            const tr = document.createElement("tr");
            tr.className = "demo-row" + (st === "ready" ? " ready" : st === "needs_embedding" ? " needs-embedding" : "");
            tr.setAttribute("role", "button");
            tr.setAttribute("tabindex", "0");
            tr.setAttribute("aria-label", `${ds.label}: ${ds.description || ""}`);
            tr.onclick = () => loadDemo(ds.name);
            tr.addEventListener("keydown", (e) => {
              if (e.key === "Enter" || e.key === " ") { e.preventDefault(); loadDemo(ds.name); }
            });
            const desc = ds.description || "";
            const descShort = desc.length > 60 ? desc.slice(0, 57) + "…" : desc;
            tr.innerHTML = `
              <td class="col-name">${escapeHtml(ds.label)}</td>
              <td class="col-num">${ds.num_files}</td>
              <td class="col-num">${ds.num_categories}</td>
              <td class="col-desc" title="${escapeHtml(desc)}">${escapeHtml(descShort)}</td>
              <td class="col-status">${buildStatusBadge(st)}</td>
            `;
            tbody.appendChild(tr);
          });

          // Update header sort indicators
          section.querySelectorAll("th[data-sort]").forEach(th => {
            const arrow = th.querySelector(".sort-arrow");
            if (th.dataset.sort === sortState.key) {
              arrow.textContent = sortState.asc ? " ▲" : " ▼";
            } else {
              arrow.textContent = "";
            }
          });
        }

        // Build tab bar and content sections
        const availableTypes = mediaOrder;

        // Fetch the user's autoload media types to pick the initial tab
        let initialTab = availableTypes[0] || Object.keys(mediaTypesMap)[0] || "audio";
        try {
          const settingsRes = await fetch("/api/settings");
          if (settingsRes.ok) {
            const settingsData = await settingsRes.json();
            const favs = settingsData.autoload_media_types || [];
            const firstFav = favs.find(f => availableTypes.includes(f));
            if (firstFav) {
              initialTab = firstFav;
            }
          }
        } catch (_) { /* ignore – just use first available tab */ }

        const tabBar = document.createElement("div");
        tabBar.className = "demo-tab-bar";
        demoDatasetsDiv.appendChild(tabBar);

        const demoContentArea = document.createElement("div");
        demoContentArea.className = "demo-content-area";
        demoDatasetsDiv.appendChild(demoContentArea);

        const sections = {};

        availableTypes.forEach(mt => {
          const items = grouped[mt];
          const mtInfo = mediaTypesMap[mt] || { icon: "📁", tab_title: mt };

          // Create tab button
          const tab = document.createElement("button");
          tab.className = "demo-tab";
          tab.dataset.mediaType = mt;
          tab.textContent = `${mtInfo.icon} ${mtInfo.tab_title}`;
          tab.addEventListener("click", () => {
            // Activate this tab, deactivate others
            tabBar.querySelectorAll(".demo-tab").forEach(t => t.classList.remove("active"));
            tab.classList.add("active");
            Object.keys(sections).forEach(k => {
              sections[k].style.display = k === mt ? "" : "none";
            });
          });
          tabBar.appendChild(tab);

          // Create section (table)
          const section = document.createElement("div");
          section.className = "demo-section";
          section._demoSort = { key: "num_files", asc: true };
          section.style.display = "none";

          const headerRow = sortColumns.map(col =>
            `<th data-sort="${col.key}">${col.label}<span class="sort-arrow"></span></th>`
          ).join("");

          section.innerHTML = `
            <table class="demo-table">
              <thead><tr>${headerRow}</tr></thead>
              <tbody></tbody>
            </table>
          `;

          // Wire up column header click sorting
          section.querySelectorAll("th[data-sort]").forEach(th => {
            th.addEventListener("click", () => {
              const key = th.dataset.sort;
              if (section._demoSort.key === key) {
                section._demoSort.asc = !section._demoSort.asc;
              } else {
                section._demoSort = { key, asc: true };
              }
              renderTable(items, section);
            });
          });

          renderTable(items, section);
          sections[mt] = section;
          demoContentArea.appendChild(section);
        });

        // Activate the initial tab
        const initialTabBtn = tabBar.querySelector(`.demo-tab[data-media-type="${initialTab}"]`);
        if (initialTabBtn) {
          initialTabBtn.classList.add("active");
          sections[initialTab].style.display = "";
        }
      } catch (e) {
        demoDatasetsDiv.innerHTML = `<div style="color:var(--color-bad); text-align:center;">Error loading demo datasets: ${escapeHtml(e.message)}</div>`;
      }
    });
    datasetLoadColumn.appendChild(demoBtnEl);

  }

  function showExtendedImporterForm(importer) {
    datasetOptions.style.display = "none";
    backButton.style.display = "block";

    let html = `<div class="form-container">`;
    html += `<h3 class="form-heading">${escapeHtml(importer.display_name)}</h3>`;
    html += `<form id="ext-imp-form">`;
    for (const field of importer.fields) {
      html += `<div class="form-group">`;
      html += `<label class="form-label">${escapeHtml(field.label)}${field.required ? " *" : ""}</label>`;
      if (field.field_type === "file") {
        html += `<input type="file" name="${escapeHtml(field.key)}" accept="${escapeHtml(field.accept)}" class="form-input" ${field.required ? "required" : ""}>`;
      } else if (field.field_type === "select") {
        html += `<select name="${escapeHtml(field.key)}" class="form-input">`;
        for (const opt of field.options) {
          html += `<option value="${escapeHtml(opt)}"${opt === field.default ? " selected" : ""}>${escapeHtml(opt)}</option>`;
        }
        html += `</select>`;
      } else if (field.field_type === "folder") {
        html += `<div class="form-row">`;
        html += `<input type="text" name="${escapeHtml(field.key)}" placeholder="${escapeHtml(field.description)}" class="form-input" style="flex:1;" data-folder-input="true" ${field.required ? "required" : ""}>`;
        html += `<button type="button" data-browse-btn="true" class="btn-browse">Browse…</button>`;
        html += `</div>`;
        html += `<input type="file" data-folder-picker="true" webkitdirectory style="display:none;">`;
      } else {
        const itype = field.field_type === "url" ? "url" : "text";
        html += `<input type="${itype}" name="${escapeHtml(field.key)}" value="${escapeHtml(field.default)}" placeholder="${escapeHtml(field.description)}" class="form-input" ${field.required ? "required" : ""}>`;
      }
      if (field.description) {
        html += `<div class="form-hint">${escapeHtml(field.description)}</div>`;
      }
      html += `</div>`;
    }
    html += `<button type="submit" class="btn-block-primary">Import</button>`;
    html += `</form></div>`;

    extendedImporterForm.innerHTML = html;
    extendedImporterForm.style.display = "block";

    // Wire up folder browse buttons
    const browseBtn = extendedImporterForm.querySelector("[data-browse-btn]");
    const folderPicker = extendedImporterForm.querySelector("[data-folder-picker]");
    const folderTextInput = extendedImporterForm.querySelector("[data-folder-input]");
    if (browseBtn && folderPicker && folderTextInput) {
      browseBtn.addEventListener("click", () => folderPicker.click());
      folderPicker.addEventListener("change", () => {
        if (folderPicker.files.length > 0) {
          // webkitRelativePath is "folderName/sub/file" — top segment is the folder name
          const topFolder = folderPicker.files[0].webkitRelativePath.split("/")[0];
          if (!folderTextInput.value) {
            folderTextInput.placeholder = `Selected: ${topFolder} — enter full path below`;
          }
        }
      });
    }

    document.getElementById("ext-imp-form").addEventListener("submit", async (e) => {
      e.preventDefault();
      const formEl = e.target;
      const hasFiles = importer.fields.some(f => f.field_type === "file");
      let body, headers = {};
      if (hasFiles) {
        body = new FormData(formEl);
      } else {
        const obj = {};
        for (const field of importer.fields) {
          obj[field.key] = formEl.elements[field.key].value;
        }
        body = JSON.stringify(obj);
        headers["Content-Type"] = "application/json";
      }
      startProgressPolling();
      try {
        const res = await fetch(`/api/dataset/import/${importer.name}`, { method: "POST", headers, body });
        if (!res.ok) {
          const err = await res.json();
          progressMessage.textContent = `Error: ${err.error}`;
          progressMessage.style.color = "var(--color-bad)";
          stopProgressPolling();
        }
      } catch (err) {
        progressMessage.textContent = `Error: ${err.message}`;
        progressMessage.style.color = "var(--color-bad)";
        stopProgressPolling();
      }
    });
  }

  // ---- Combine Existing Datasets UI ----

  async function showCombineDatasetsForm() {
    // Initialise state on first entry; preserved across re-renders.
    if (!_combineState) _combineState = [];

    datasetOptions.style.display = "none";
    datasetProgress.style.display = "none";
    demoDatasetsDiv.style.display = "none";
    backButton.style.display = "block";

    const staged = _combineState;

    // --- Build the HTML ---
    let html = `<div class="form-container wide">`;
    html += `<h3 class="form-heading compact">\uD83D\uDD00 Combine Datasets</h3>`;
    html += `<p class="form-text" style="margin-bottom:16px;">Add two or more datasets, then combine them. All must be the same media type. Duplicates are skipped automatically.</p>`;

    // Staged datasets list
    html += `<div id="combine-staged-list" style="margin-bottom:16px;">`;
    if (staged.length === 0) {
      html += `<p class="form-text">No datasets added yet. Use the buttons below to add datasets.</p>`;
    } else {
      staged.forEach((ds, i) => {
        html += `<div class="combine-item">`;
        html += `<span class="combine-item-name">${escapeHtml(ds.name)}</span>`;
        html += `<span class="combine-item-count">${ds.count} medias</span>`;
        html += `<button type="button" data-combine-remove="${i}" class="combine-item-remove" title="Remove">&times;</button>`;
        html += `</div>`;
      });
    }
    html += `</div>`;

    // "Add dataset" section – same buttons as the welcome screen
    html += `<div style="margin-bottom:16px;">`;
    html += `<div style="font-size:0.85rem;color:var(--text-secondary);margin-bottom:8px;font-weight:600;">Add a dataset:</div>`;
    html += `<div class="dataset-columns" id="combine-add-columns" style="gap:12px;">`;
    html += `<div class="dataset-column" id="combine-load-column">`;
    html += `<div class="dataset-column-header">Load</div>`;
    html += `<button class="dataset-option" id="combine-add-file-btn"><h3>\uD83D\uDCC1 Load from File</h3><p>Upload a saved dataset file (.pkl)</p></button>`;
    html += `</div>`;
    html += `<div class="dataset-column" id="combine-generate-column">`;
    html += `<div class="dataset-column-header">Generate</div>`;
    html += `</div>`;
    html += `</div></div>`;

    // Combine button
    html += `<button type="button" id="combine-submit-btn" class="btn-block-primary" disabled>Add at least 2 datasets</button>`;
    html += `</div>`;

    extendedImporterForm.innerHTML = html;
    extendedImporterForm.style.display = "block";

    // --- Wire up remove buttons ---
    extendedImporterForm.querySelectorAll("[data-combine-remove]").forEach(btn => {
      btn.addEventListener("click", () => {
        staged.splice(parseInt(btn.dataset.combineRemove), 1);
        showCombineDatasetsForm();
      });
    });

    // --- Wire up "Add from File" button ---
    const combineFileInput = document.createElement("input");
    combineFileInput.type = "file";
    combineFileInput.accept = ".pkl";
    combineFileInput.style.display = "none";
    extendedImporterForm.appendChild(combineFileInput);

    document.getElementById("combine-add-file-btn").addEventListener("click", () => {
      combineFileInput.click();
    });
    combineFileInput.addEventListener("change", async () => {
      const file = combineFileInput.files[0];
      if (!file) return;
      const formData = new FormData();
      formData.append("file", file);
      try {
        const res = await fetch("/api/dataset/stage-file", { method: "POST", body: formData });
        if (!res.ok) {
          const err = await res.json();
          vtAlert(`Error: ${err.error}`, "warning");
          return;
        }
        const result = await res.json();
        staged.push(result);
        showCombineDatasetsForm();
      } catch (err) {
        vtAlert(`Error: ${err.message}`, "warning");
      }
      combineFileInput.value = "";
    });

    // --- Populate extended importer buttons ---
    const combineGenCol = document.getElementById("combine-generate-column");
    const combineLoadCol = document.getElementById("combine-load-column");
    try {
      const res = await fetch("/api/dataset/importers");
      if (res.ok) {
        const data = await res.json();
        for (const imp of data.importers) {
          const btn = document.createElement("button");
          btn.className = "dataset-option";
          btn.innerHTML = `<h3>${escapeHtml(imp.icon || "\uD83D\uDD0C")} ${escapeHtml(imp.display_name)}</h3><p>${escapeHtml(imp.description)}</p>`;
          btn.addEventListener("click", () => showCombineStagingImporterForm(imp));
          combineGenCol.appendChild(btn);
        }
      }
    } catch (_) { /* extended importers are optional */ }

    // "Add Demo Dataset" button
    const demoBtn = document.createElement("button");
    demoBtn.className = "dataset-option";
    demoBtn.innerHTML = `<h3>\uD83C\uDFC6 Load Demo Dataset</h3><p>Stage a demo dataset for combining</p>`;
    demoBtn.addEventListener("click", () => showCombineStagingDemoList());
    combineLoadCol.appendChild(demoBtn);

    // --- Update submit button state ---
    const submitBtn = document.getElementById("combine-submit-btn");
    if (staged.length >= 2) {
      submitBtn.disabled = false;
      submitBtn.textContent = `Combine ${staged.length} Datasets`;
    }

    // --- Handle combine submit ---
    submitBtn.addEventListener("click", async () => {
      if (staged.length < 2) return;
      const paths = staged.map(ds => ds.path);
      // Leave combine mode so the normal progress handler takes over.
      _combineState = null;
      startProgressPolling();
      try {
        const res = await fetch("/api/dataset/combine", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ datasets: paths }),
        });
        if (!res.ok) {
          const err = await res.json();
          progressMessage.textContent = `Error: ${err.error}`;
          progressMessage.style.color = "var(--color-bad)";
          stopProgressPolling();
        }
      } catch (err) {
        progressMessage.textContent = `Error: ${err.message}`;
        progressMessage.style.color = "var(--color-bad)";
        stopProgressPolling();
      }
      // Clean up staging files in the background.
      fetch("/api/dataset/staging", { method: "DELETE" }).catch(() => {});
    });
  }

  // Show an extended importer form inside the combine flow (staging mode).
  function showCombineStagingImporterForm(importer) {
    datasetOptions.style.display = "none";
    extendedImporterForm.style.display = "block";
    backButton.style.display = "block";

    let html = `<div class="form-container">`;
    html += `<h3 class="form-heading">${escapeHtml(importer.display_name)}</h3>`;
    html += `<p class="form-text">This will be added to the combine list.</p>`;
    html += `<form id="combine-stage-form">`;
    for (const field of importer.fields) {
      html += `<div class="form-group">`;
      html += `<label class="form-label">${escapeHtml(field.label)}${field.required ? " *" : ""}</label>`;
      if (field.field_type === "file") {
        html += `<input type="file" name="${escapeHtml(field.key)}" accept="${escapeHtml(field.accept)}" class="form-input" ${field.required ? "required" : ""}>`;
      } else if (field.field_type === "select") {
        html += `<select name="${escapeHtml(field.key)}" class="form-input">`;
        for (const opt of field.options) {
          html += `<option value="${escapeHtml(opt)}"${opt === field.default ? " selected" : ""}>${escapeHtml(opt)}</option>`;
        }
        html += `</select>`;
      } else if (field.field_type === "folder") {
        html += `<div class="form-row">`;
        html += `<input type="text" name="${escapeHtml(field.key)}" placeholder="${escapeHtml(field.description)}" class="form-input" style="flex:1;" data-folder-input="true" ${field.required ? "required" : ""}>`;
        html += `<button type="button" data-browse-btn="true" class="btn-browse">Browse\u2026</button>`;
        html += `</div>`;
        html += `<input type="file" data-folder-picker="true" webkitdirectory style="display:none;">`;
      } else {
        const itype = field.field_type === "url" ? "url" : "text";
        html += `<input type="${itype}" name="${escapeHtml(field.key)}" value="${escapeHtml(field.default)}" placeholder="${escapeHtml(field.description)}" class="form-input" ${field.required ? "required" : ""}>`;
      }
      if (field.description) {
        html += `<div class="form-hint">${escapeHtml(field.description)}</div>`;
      }
      html += `</div>`;
    }
    html += `<button type="submit" class="btn-block-primary">Add to combine list</button>`;
    html += `</form></div>`;

    extendedImporterForm.innerHTML = html;

    // Wire up folder browse buttons
    const browseBtn = extendedImporterForm.querySelector("[data-browse-btn]");
    const folderPicker = extendedImporterForm.querySelector("[data-folder-picker]");
    const folderTextInput = extendedImporterForm.querySelector("[data-folder-input]");
    if (browseBtn && folderPicker && folderTextInput) {
      browseBtn.addEventListener("click", () => folderPicker.click());
      folderPicker.addEventListener("change", () => {
        if (folderPicker.files.length > 0) {
          const topFolder = folderPicker.files[0].webkitRelativePath.split("/")[0];
          if (!folderTextInput.value) {
            folderTextInput.placeholder = `Selected: ${topFolder} \u2014 enter full path below`;
          }
        }
      });
    }

    document.getElementById("combine-stage-form").addEventListener("submit", async (e) => {
      e.preventDefault();
      const formEl = e.target;
      const hasFiles = importer.fields.some(f => f.field_type === "file");
      let body, headers = {};
      if (hasFiles) {
        body = new FormData(formEl);
      } else {
        const obj = {};
        for (const field of importer.fields) {
          obj[field.key] = formEl.elements[field.key].value;
        }
        body = JSON.stringify(obj);
        headers["Content-Type"] = "application/json";
      }
      startProgressPolling();
      try {
        const res = await fetch(`/api/dataset/stage-import/${importer.name}`, { method: "POST", headers, body });
        if (!res.ok) {
          const err = await res.json();
          progressMessage.textContent = `Error: ${err.error}`;
          progressMessage.style.color = "var(--color-bad)";
          stopProgressPolling();
        }
      } catch (err) {
        progressMessage.textContent = `Error: ${err.message}`;
        progressMessage.style.color = "var(--color-bad)";
        stopProgressPolling();
      }
    });
  }

  // Show the demo dataset list inside the combine flow (staging mode).
  async function showCombineStagingDemoList() {
    datasetOptions.style.display = "none";
    extendedImporterForm.style.display = "block";
    backButton.style.display = "block";

    let html = `<div class="form-container">`;
    html += `<h3 class="form-heading">\uD83C\uDFC6 Add Demo Dataset</h3>`;
    html += `<p class="form-text">Select a demo dataset to add to the combine list.</p>`;
    html += `<div id="combine-demo-list"><p style="color:var(--text-dim);">Loading...</p></div>`;
    html += `</div>`;
    extendedImporterForm.innerHTML = html;

    const listDiv = document.getElementById("combine-demo-list");
    try {
      const res = await fetch("/api/dataset/demo-list");
      if (!res.ok) throw new Error("Failed to load");
      const data = await res.json();
      let listHtml = "";
      for (const ds of data.datasets) {
        listHtml += `<button class="dataset-option" data-demo-name="${escapeHtml(ds.name)}" style="width:100%;text-align:left;margin-bottom:6px;">`;
        listHtml += `<h3>${escapeHtml(ds.label)}</h3>`;
        listHtml += `<p>${escapeHtml(ds.description)} (${ds.num_files} files)</p>`;
        listHtml += `</button>`;
      }
      listDiv.innerHTML = listHtml || `<p style="color:var(--text-dim);">No demo datasets available.</p>`;
      listDiv.querySelectorAll("[data-demo-name]").forEach(btn => {
        btn.addEventListener("click", async () => {
          const name = btn.dataset.demoName;
          startProgressPolling();
          try {
            const res = await fetch(`/api/dataset/stage-demo/${encodeURIComponent(name)}`, { method: "POST" });
            if (!res.ok) {
              const err = await res.json();
              progressMessage.textContent = `Error: ${err.error}`;
              progressMessage.style.color = "var(--color-bad)";
              stopProgressPolling();
            }
          } catch (err) {
            progressMessage.textContent = `Error: ${err.message}`;
            progressMessage.style.color = "var(--color-bad)";
            stopProgressPolling();
          }
        });
      });
    } catch (e) {
      listDiv.innerHTML = `<p style="color:var(--color-bad);">Error loading demos: ${escapeHtml(e.message)}</p>`;
    }
  }

  loadExtendedImporters();

  backButton.addEventListener("click", async () => {
    // If we are inside a staging sub-form (importer or demo) within the
    // combine flow, go back to the combine form instead of the welcome screen.
    const stageForm = document.getElementById("combine-stage-form");
    const demoList = document.getElementById("combine-demo-list");
    if (_combineState && (stageForm || demoList)) {
      showCombineDatasetsForm();
      return;
    }
    // Leaving the combine flow entirely – clean up state.
    if (_combineState) {
      fetch("/api/dataset/staging", { method: "DELETE" }).catch(() => {});
      _combineState = null;
    }
    // If we came from the dashboard, go back to dashboard
    if (_dashboardAddDatasetMode) {
      _dashboardAddDatasetMode = false;
      showDashboard();
      return;
    }
    // If we were in a training session, save labels and return to dashboard
    if (_dashboardTrainMode) {
      await saveTrainableModelLabels();
      showDashboard();
      return;
    }
    showWelcomeScreen();
  });

  // ---- Burger Menu ----

  // Pause/resume looping media when focus leaves the labeling interface
  function pauseActiveMedia() {
    const audio = document.getElementById("media-audio");
    const video = document.getElementById("media-video");
    window._mediaPausedForUI = false;
    if (audio && !audio.paused) { audio.pause(); window._mediaPausedForUI = true; }
    if (video && !video.paused) { video.pause(); window._mediaPausedForUI = true; }
  }

  function resumeActiveMedia() {
    if (!window._mediaPausedForUI) return;
    const audio = document.getElementById("media-audio");
    const video = document.getElementById("media-video");
    if (audio) audio.play().catch(() => {});
    if (video) video.play().catch(() => {});
    window._mediaPausedForUI = false;
  }

  // Toggle burger menu
  function closeBurgerMenu() {
    burgerDropdown.classList.remove("show");
    burgerBtn.setAttribute("aria-expanded", "false");
    resumeActiveMedia();
  }

  function openBurgerMenu() {
    burgerDropdown.classList.add("show");
    burgerBtn.setAttribute("aria-expanded", "true");
    pauseActiveMedia();
    // Focus first menu item
    const firstItem = burgerDropdown.querySelector('[role="menuitem"]');
    if (firstItem) firstItem.focus();
  }

  if (burgerBtn && burgerDropdown) {
    burgerBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      if (burgerDropdown.classList.contains("show")) {
        closeBurgerMenu();
      } else {
        openBurgerMenu();
      }
    });

    // Keyboard navigation within burger menu
    burgerDropdown.addEventListener("keydown", (e) => {
      const items = Array.from(burgerDropdown.querySelectorAll('[role="menuitem"]'));
      const currentIndex = items.indexOf(document.activeElement);

      switch (e.key) {
        case "ArrowDown":
          e.preventDefault();
          if (currentIndex < items.length - 1) items[currentIndex + 1].focus();
          else items[0].focus();
          break;
        case "ArrowUp":
          e.preventDefault();
          if (currentIndex > 0) items[currentIndex - 1].focus();
          else items[items.length - 1].focus();
          break;
        case "Escape":
          e.preventDefault();
          closeBurgerMenu();
          burgerBtn.focus();
          break;
        case "Enter":
        case " ":
          if (document.activeElement.getAttribute("role") === "menuitem") {
            e.preventDefault();
            document.activeElement.click();
          }
          break;
        case "Home":
          e.preventDefault();
          items[0].focus();
          break;
        case "End":
          e.preventDefault();
          items[items.length - 1].focus();
          break;
      }
    });

    // Close burger menu when clicking outside
    document.addEventListener("click", (e) => {
      if (!burgerDropdown.contains(e.target) && !burgerBtn.contains(e.target)) {
        if (burgerDropdown.classList.contains("show")) {
          closeBurgerMenu();
        }
      }
    });
  }

  // Labels export – open modal (used by detector export modal)
  // Options: { goodsOnly: bool } — when true, only export "good" labels
  async function openLabelExporterModal(options) {
    const goodsOnly = options && options.goodsOnly;
    let exporters = [];
    try {
      const res = await fetch("/api/exporters");
      if (res.ok) exporters = await res.json();
    } catch (_) { /* ignore */ }

    if (exporters.length === 0) {
      labelExporterList.innerHTML = '<p style="color:var(--text-muted);">No label exporters available.</p>';
    } else {
      labelExporterList.innerHTML = exporters.map(exp => `
        <div class="label-exporter-option option-card" data-name="${escapeHtml(exp.name)}">
          <span class="option-card-icon">${escapeHtml(exp.icon || '\uD83D\uDCE4')}</span>
          <div>
            <div class="option-card-title">${escapeHtml(exp.display_name)}</div>
            <div class="option-card-desc">${escapeHtml(exp.description)}</div>
          </div>
        </div>
      `).join("");

      labelExporterList.querySelectorAll(".label-exporter-option").forEach(el => {
        el.setAttribute("role", "button");
        el.setAttribute("tabindex", "0");
        const name = el.dataset.name;
        const exp = exporters.find(e => e.name === name);
        el.addEventListener("click", () => {
          labelExporterModal.classList.remove("show");
          runLabelExport(exp, { goodsOnly });
        });
        el.addEventListener("keydown", (e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            labelExporterModal.classList.remove("show");
            runLabelExport(exp, { goodsOnly });
          }
        });
      });
    }

    // Update modal title to reflect what's being exported
    const titleEl = document.getElementById("label-exporter-modal-title");
    if (titleEl) titleEl.textContent = goodsOnly ? "Export Labels (Goods)" : "Export Labels";

    labelExporterModal.classList.add("show");
  }

  async function runLabelExport(exp, options) {
    menuLabelsStatus.textContent = "";
    const goodsOnly = options && options.goodsOnly;
    // Collect required field values via prompts
    const fieldValues = {};
    for (const field of exp.fields) {
      if (field.required) {
        const val = await vtPrompt(field.label + (field.description ? ` (${field.description})` : ""), field.default || "");
        if (val === null) {
          menuLabelsStatus.textContent = "Export cancelled";
          setTimeout(() => { menuLabelsStatus.textContent = ""; }, 2000);
          return;
        }
        fieldValues[field.key] = val;
      } else {
        fieldValues[field.key] = field.default || "";
      }
    }

    menuLabelsStatus.textContent = "Exporting labels\u2026";
    try {
      const labelsUrl = goodsOnly ? "/api/labels/export?goods_only=1" : "/api/labels/export";
      const labelsRes = await fetch(labelsUrl);
      const labelsData = await labelsRes.json();

      const exportRes = await fetch("/api/exporters/export", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          exporter_name: exp.name,
          field_values: fieldValues,
          results: labelsData,
        }),
      });
      const result = await exportRes.json();
      if (!exportRes.ok) {
        menuLabelsStatus.textContent = result.error || "Export failed";
        setTimeout(() => { menuLabelsStatus.textContent = ""; }, 3000);
        return;
      }
      // Handle browser download if the exporter returns download_content
      if (result.download_content) {
        const blob = new Blob([result.download_content], { type: result.download_content_type || "application/json" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = result.download_filename || "labels.json";
        a.click();
        URL.revokeObjectURL(url);
      }
      menuLabelsStatus.textContent = result.message || "Labels exported";
      setTimeout(() => { menuLabelsStatus.textContent = ""; }, 4000);
    } catch (e) {
      menuLabelsStatus.textContent = "Export failed";
      setTimeout(() => { menuLabelsStatus.textContent = ""; }, 3000);
    }
  }

  // Labels import – open the label importer picker modal
  if (menuLabelsImport && burgerDropdown) {
    menuLabelsImport.addEventListener("click", async () => {
      if (menuLabelsImport.classList.contains("disabled")) return;
      closeBurgerMenu();
      await openLabelImporterModal();
    });
  }

  // Detector import — open Load Sort modal
  if (menuDetectorImport && burgerDropdown) {
    menuDetectorImport.addEventListener("click", () => {
      if (menuDetectorImport.classList.contains("disabled")) return;
      closeBurgerMenu();
      openLoadSortModal();
    });
  }

  async function openDetectorExportModal(detectorName) {
    // Update modal title
    const titleEl = document.getElementById("detector-export-modal-title");
    if (titleEl) titleEl.textContent = detectorName ? `Export "${detectorName}"` : "Export Detector";

    // Fetch labelset exporters in parallel with building the modal
    let labelsetExporters = [];
    try {
      const res = await fetch("/api/exporters");
      if (res.ok) labelsetExporters = await res.json();
    } catch (_) { /* ignore */ }
    // Filter out the GUI exporter (not useful for file-based export)
    labelsetExporters = labelsetExporters.filter(e => e.name !== "gui");

    let html = `
      <div id="detector-export-browser-btn" class="option-card" role="button" tabindex="0">
        <span class="option-card-icon">\u2B07\uFE0F</span>
        <div>
          <div class="option-card-title">Download (Browser)</div>
          <div class="option-card-desc">Download the detector file directly to your browser.</div>
        </div>
      </div>
      <div id="detector-export-server-btn" class="option-card" role="button" tabindex="0">
        <span class="option-card-icon">\uD83D\uDCBE</span>
        <div>
          <div class="option-card-title">Save to Server</div>
          <div class="option-card-desc">Save the detector file to the server disk.</div>
        </div>
      </div>
    `;

    if (labelsetExporters.length > 0) {
      html += `<h3 style="margin:1.2em 0 0.3em;font-size:1em;color:var(--text-muted);">Export Labels</h3>
        <p style="margin:0 0 0.6em;font-size:0.85em;color:var(--text-muted);">Export votes as a label set (sufficient to retrain the detector later).</p>`;
      html += labelsetExporters.map(exp => `
        <div class="detector-labelset-export-option option-card" data-name="${escapeHtml(exp.name)}" role="button" tabindex="0">
          <span class="option-card-icon">${escapeHtml(exp.icon || '\uD83D\uDCE4')}</span>
          <div>
            <div class="option-card-title">${escapeHtml(exp.display_name)}</div>
            <div class="option-card-desc">${escapeHtml(exp.description)}</div>
          </div>
        </div>
      `).join("");
    }

    detectorExportList.innerHTML = html;

    const browserBtn = detectorExportList.querySelector("#detector-export-browser-btn");
    const serverBtn = detectorExportList.querySelector("#detector-export-server-btn");

    browserBtn.addEventListener("click", () => { detectorExportModal.classList.remove("show"); runDetectorExportBrowser(detectorName); });
    browserBtn.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); detectorExportModal.classList.remove("show"); runDetectorExportBrowser(detectorName); }
    });
    serverBtn.addEventListener("click", () => { detectorExportModal.classList.remove("show"); runDetectorExportServer(detectorName); });
    serverBtn.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); detectorExportModal.classList.remove("show"); runDetectorExportServer(detectorName); }
    });

    // Wire up labelset exporter options
    detectorExportList.querySelectorAll(".detector-labelset-export-option").forEach(el => {
      const name = el.dataset.name;
      const exp = labelsetExporters.find(e => e.name === name);
      el.addEventListener("click", () => {
        detectorExportModal.classList.remove("show");
        runDetectorLabelExport(exp);
      });
      el.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          detectorExportModal.classList.remove("show");
          runDetectorLabelExport(exp);
        }
      });
    });

    detectorExportModal.classList.add("show");
  }

  async function runDetectorLabelExport(exp) {
    if (menuDetectorStatus) menuDetectorStatus.textContent = "";
    // Collect required field values via prompts
    const fieldValues = {};
    for (const field of exp.fields) {
      if (field.required) {
        const val = await vtPrompt(field.label + (field.description ? ` (${field.description})` : ""), field.default || "");
        if (val === null) {
          if (menuDetectorStatus) { menuDetectorStatus.textContent = "Export cancelled"; setTimeout(() => { menuDetectorStatus.textContent = ""; }, 2000); }
          return;
        }
        fieldValues[field.key] = val;
      } else {
        fieldValues[field.key] = field.default || "";
      }
    }

    if (menuDetectorStatus) menuDetectorStatus.textContent = "Exporting labels\u2026";
    try {
      const labelsRes = await fetch("/api/labels/export");
      const labelsData = await labelsRes.json();

      const exportRes = await fetch("/api/exporters/export", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          exporter_name: exp.name,
          field_values: fieldValues,
          results: labelsData,
        }),
      });
      const result = await exportRes.json();
      if (!exportRes.ok) {
        if (menuDetectorStatus) { menuDetectorStatus.textContent = result.error || "Export failed"; setTimeout(() => { menuDetectorStatus.textContent = ""; }, 3000); }
        return;
      }
      if (result.download_content) {
        const blob = new Blob([result.download_content], { type: result.download_content_type || "application/json" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = result.download_filename || "labels.json";
        a.click();
        URL.revokeObjectURL(url);
      }
      if (menuDetectorStatus) { menuDetectorStatus.textContent = result.message || "Labels exported"; setTimeout(() => { menuDetectorStatus.textContent = ""; }, 4000); }
    } catch (e) {
      if (menuDetectorStatus) { menuDetectorStatus.textContent = "Export failed"; setTimeout(() => { menuDetectorStatus.textContent = ""; }, 3000); }
    }
  }

  async function runDetectorExportBrowser(detectorName) {
    if (menuDetectorStatus) menuDetectorStatus.textContent = "Exporting detector\u2026";
    try {
      const res = await fetch(`/api/autorun-detectors/${encodeURIComponent(detectorName)}/export`);
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        if (menuDetectorStatus) { menuDetectorStatus.textContent = err.error || "Export failed"; setTimeout(() => { menuDetectorStatus.textContent = ""; }, 3000); }
        return;
      }
      const data = await res.json();
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${detectorName}.json`;
      a.click();
      URL.revokeObjectURL(url);
      if (menuDetectorStatus) { menuDetectorStatus.textContent = "Detector exported (browser)"; setTimeout(() => { menuDetectorStatus.textContent = ""; }, 3000); }
    } catch (_) {
      if (menuDetectorStatus) { menuDetectorStatus.textContent = "Export failed"; setTimeout(() => { menuDetectorStatus.textContent = ""; }, 3000); }
    }
  }

  async function runDetectorExportServer(detectorName) {
    const filename = await vtPrompt("Enter a name for the detector file (saved on server):", detectorName);
    if (!filename || !filename.trim()) return;

    if (menuDetectorStatus) menuDetectorStatus.textContent = "Saving detector to server\u2026";

    const res = await fetch(`/api/autorun-detectors/${encodeURIComponent(detectorName)}/export-server`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ filename: filename.trim() }),
    });

    if (res.status === 409) {
      const info = await res.json();
      const overwrite = await vtConfirm(
        `A detector file "${info.name}.json" already exists on the server.\n\nPath: ${info.path}\n\nOverwrite it?`
      );
      if (overwrite) {
        if (menuDetectorStatus) menuDetectorStatus.textContent = "Overwriting detector on server\u2026";
        const res2 = await fetch(`/api/autorun-detectors/${encodeURIComponent(detectorName)}/export-server`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ filename: filename.trim(), overwrite: true }),
        });
        if (res2.ok) {
          const data2 = await res2.json();
          if (menuDetectorStatus) { menuDetectorStatus.textContent = `Saved to server: ${data2.name}.json`; setTimeout(() => { menuDetectorStatus.textContent = ""; }, 4000); }
        } else {
          const err = await res2.json().catch(() => ({}));
          if (menuDetectorStatus) { menuDetectorStatus.textContent = err.error || "Server export failed"; setTimeout(() => { menuDetectorStatus.textContent = ""; }, 3000); }
        }
      } else {
        const newName = await vtPrompt("Enter a different name:");
        if (newName && newName.trim()) {
          if (menuDetectorStatus) menuDetectorStatus.textContent = "Saving detector to server\u2026";
          const res3 = await fetch(`/api/autorun-detectors/${encodeURIComponent(detectorName)}/export-server`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ filename: newName.trim() }),
          });
          if (res3.ok) {
            const data3 = await res3.json();
            if (menuDetectorStatus) { menuDetectorStatus.textContent = `Saved to server: ${data3.name}.json`; setTimeout(() => { menuDetectorStatus.textContent = ""; }, 4000); }
          } else {
            const err = await res3.json().catch(() => ({}));
            if (menuDetectorStatus) { menuDetectorStatus.textContent = err.error || "Server export failed"; setTimeout(() => { menuDetectorStatus.textContent = ""; }, 3000); }
          }
        } else {
          if (menuDetectorStatus) { menuDetectorStatus.textContent = "Export cancelled"; setTimeout(() => { menuDetectorStatus.textContent = ""; }, 2000); }
        }
      }
    } else if (res.ok) {
      const data = await res.json();
      if (menuDetectorStatus) { menuDetectorStatus.textContent = `Saved to server: ${data.name}.json`; setTimeout(() => { menuDetectorStatus.textContent = ""; }, 4000); }
    } else {
      const err = await res.json().catch(() => ({}));
      if (menuDetectorStatus) { menuDetectorStatus.textContent = err.error || "Server export failed"; setTimeout(() => { menuDetectorStatus.textContent = ""; }, 3000); }
    }
  }

  // Train-context-bar export buttons
  if (trainExportDetectorBtn) {
    trainExportDetectorBtn.addEventListener("click", () => {
      const name = trainDetectorName ? trainDetectorName.textContent : "";
      if (name) openDetectorExportModal(name);
    });
  }
  if (trainExportLabelsBtn) {
    trainExportLabelsBtn.addEventListener("click", () => {
      openLabelExporterModal({ goodsOnly: true });
    });
  }

  // Results display, export controls, escapeHtml, formatOrigin
  // delegated to static/results.js (window.VTResults)
  const escapeHtml = window.VTResults.escapeHtml;
  const formatOrigin = window.VTResults.formatOrigin;
  const displayAutodetectResults = window.VTResults.displayAutodetectResults;
  const displayFindResults = window.VTResults.displayFindResults;
  window.VTResults.init({
    getSortOrder: () => sortOrder,
    getThreshold: () => threshold,
    getVotes: () => votes,
    getGoodVoteSet: () => goodVoteSet,
    getBadVoteSet: () => badVoteSet,
    setVotes: (v) => { votes = v; _rebuildVoteSets(); },
    renderVotes: () => renderVotes(),
  });

  // ---- Settings modal ----

  function populateSettingsModal(data) {
    applyTheme(data.theme || "dark");
    if (calibrateCountInput) calibrateCountInput.value = data.calibrate_count;
    if (calibrationFractionInput) calibrationFractionInput.value = data.calibration_fraction;
    if (safeThresholdsCheckbox) safeThresholdsCheckbox.checked = !!data.safe_thresholds;
    if (enrichDescCheckbox) enrichDescCheckbox.checked = !!data.enrich_descriptions;
    if (swipeAnimationCheckbox) {
      const val = data.swipe_animation !== undefined ? !!data.swipe_animation : true;
      swipeAnimationCheckbox.checked = val;
      swipeAnimation = val;
    }
    if (showThumbnailsLeftCheckbox) {
      showThumbnailsLeftCheckbox.checked = !!data.show_thumbnails_left;
      showThumbnailsLeft = !!data.show_thumbnails_left;
    }
    if (showThumbnailsRightCheckbox) {
      const val = data.show_thumbnails_right !== undefined ? !!data.show_thumbnails_right : true;
      showThumbnailsRightCheckbox.checked = val;
      showThumbnailsRight = val;
    }
    if (autopilotTopGreensInput) autopilotTopGreensInput.value = data.autopilot_top_greens;
    if (autopilotHardRedsInput) autopilotHardRedsInput.value = data.autopilot_hard_reds;
    const favList = data.autoload_media_types || [];
    favMtCheckboxes.forEach(cb => {
      cb.checked = favList.includes(cb.dataset.mediaType);
    });
  }

  if (menuSettings && settingsModal && burgerDropdown) {
    menuSettings.addEventListener("click", async () => {
      burgerDropdown.classList.remove("show");
      try {
        const res = await fetch("/api/settings");
        if (res.ok) populateSettingsModal(await res.json());
      } catch (_) {}
      settingsModal.classList.add("show");
    });
  }

  if (settingsModalClose) {
    settingsModalClose.addEventListener("click", () => {
      settingsModal.classList.remove("show");
    });
  }

  // Safe thresholds toggle
  if (safeThresholdsCheckbox) {
    safeThresholdsCheckbox.addEventListener("change", () => {
      fetch("/api/settings", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ safe_thresholds: safeThresholdsCheckbox.checked }),
      }).catch(() => {});
    });
  }

  // Swipe Animation toggle
  if (swipeAnimationCheckbox) {
    swipeAnimationCheckbox.addEventListener("change", () => {
      swipeAnimation = swipeAnimationCheckbox.checked;
      fetch("/api/settings", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ swipe_animation: swipeAnimationCheckbox.checked }),
      }).catch(() => {});
    });
  }

  // Enrich Sort Descriptions toggle
  if (enrichDescCheckbox) {
    enrichDescCheckbox.addEventListener("change", () => {
      fetch("/api/settings", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ enrich_descriptions: enrichDescCheckbox.checked }),
      }).catch(() => {});
      // Re-trigger text sort if active
      if (sortMode === "text") {
        onTextSortInput();
      }
    });
  }

  // Show Thumbnails Left toggle
  if (showThumbnailsLeftCheckbox) {
    showThumbnailsLeftCheckbox.addEventListener("change", () => {
      showThumbnailsLeft = showThumbnailsLeftCheckbox.checked;
      fetch("/api/settings", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ show_thumbnails_left: showThumbnailsLeft }),
      }).catch(() => {});
      renderMediaList();
    });
  }

  // Show Thumbnails Right toggle
  if (showThumbnailsRightCheckbox) {
    showThumbnailsRightCheckbox.addEventListener("change", () => {
      showThumbnailsRight = showThumbnailsRightCheckbox.checked;
      fetch("/api/settings", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ show_thumbnails_right: showThumbnailsRight }),
      }).catch(() => {});
      renderVotes();
    });
  }

  // Autoload media type toggles
  favMtCheckboxes.forEach(cb => {
    cb.addEventListener("change", () => {
      const selected = [];
      favMtCheckboxes.forEach(c => { if (c.checked) selected.push(c.dataset.mediaType); });
      fetch("/api/settings", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ autoload_media_types: selected }),
      }).catch(() => {});
    });
  });

  // Default button — reset all settings to defaults
  if (settingsDefaultBtn) {
    settingsDefaultBtn.addEventListener("click", async () => {
      try {
        const res = await fetch("/api/settings/defaults");
        if (!res.ok) return;
        const defaults = await res.json();
        // Apply defaults to server
        await fetch("/api/settings", {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(defaults),
        });
        // Update modal controls
        populateSettingsModal(defaults);
        // Apply theme immediately
        applyTheme(defaults.theme || "dark");
        // Update training UI controls that live outside the modal
        if (inclusionSlider) { inclusionSlider.value = defaults.inclusion || 0; inclusionValue.textContent = defaults.inclusion || 0; inclusion = defaults.inclusion || 0; }
        audioVolume = defaults.volume != null ? defaults.volume : 1.0;
        const audioEl = document.getElementById("media-audio");
        if (audioEl) audioEl.volume = audioVolume;
        renderMediaList();
        renderVotes();
      } catch (_) {}
    });
  }

  // Import settings from file
  if (settingsImportBtn && settingsImportFile) {
    settingsImportBtn.addEventListener("click", () => settingsImportFile.click());
    settingsImportFile.addEventListener("change", async () => {
      const file = settingsImportFile.files[0];
      if (!file) return;
      try {
        const text = await file.text();
        const imported = JSON.parse(text);
        // Send all importable fields to the server
        const payload = {};
        const importableKeys = ["volume", "theme", "inclusion", "enrich_descriptions", "safe_thresholds", "calibrate_count", "calibration_fraction", "swipe_animation", "show_thumbnails_left", "show_thumbnails_right", "autopilot_top_greens", "autopilot_hard_reds"];
        for (const k of importableKeys) {
          if (k in imported) payload[k] = imported[k];
        }
        const res = await fetch("/api/settings", {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        if (res.ok) {
          const data = await res.json();
          populateSettingsModal(data);
          applyTheme(data.theme || "dark");
          if (inclusionSlider) { inclusionSlider.value = data.inclusion || 0; inclusionValue.textContent = data.inclusion || 0; inclusion = data.inclusion || 0; }
          audioVolume = typeof data.volume === "number" ? data.volume : 1.0;
          const audioEl = document.getElementById("media-audio");
          if (audioEl) audioEl.volume = audioVolume;
          renderMediaList();
          renderVotes();
        }
      } catch (_) {
        vtAlert("Failed to import settings. Make sure the file is valid JSON.", "error");
      }
      settingsImportFile.value = "";
    });
  }

  // Export settings to file
  if (settingsExportBtn) {
    settingsExportBtn.addEventListener("click", async () => {
      try {
        const res = await fetch("/api/settings");
        if (!res.ok) return;
        const data = await res.json();
        // Exclude runtime-only fields
        delete data.autorun_processors;
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = "vtsearch-settings.json";
        a.click();
        URL.revokeObjectURL(url);
      } catch (_) {}
    });
  }

  // ---- Sort mode switching ----

  function updateSortModeAvailability() {
    const hasGoodAndBad = votes.good.length > 0 && votes.bad.length > 0;
    learnedRadio.disabled = !hasGoodAndBad;
    learnedRadio.parentElement.style.opacity = hasGoodAndBad ? "1" : "0.5";
    learnedRadio.parentElement.style.cursor = hasGoodAndBad ? "pointer" : "not-allowed";

    // Load radio is always enabled - selecting it prompts for detector file
    loadRadio.disabled = false;
    loadRadio.parentElement.style.opacity = "1";
    loadRadio.parentElement.style.cursor = "pointer";

  }

  document.querySelectorAll('input[name="sort-mode"]').forEach(radio => {
    radio.addEventListener("change", () => {
      // Validate selection
      if (radio.value === "learned" && (votes.good.length === 0 || votes.bad.length === 0)) {
        sortStatus.textContent = "Vote good & bad medias first";
        // Revert to text mode
        document.querySelector('input[name="sort-mode"][value="text"]').checked = true;
        return;
      }
      if (radio.value === "load") {
        // Open the Load Sort modal to choose detector or example
        sortMode = radio.value;
        textSortWrap.style.display = "none";
        learnedSortWrap.style.display = "none";
        loadSortWrap.style.display = "";
        sortStatus.textContent = "";
        if (!loadedDetector) {
          openLoadSortModal();
        }
        return;
      }

      sortMode = radio.value;
      textSortWrap.style.display = sortMode === "text" ? "" : "none";
      learnedSortWrap.style.display = sortMode === "learned" ? "" : "none";
      loadSortWrap.style.display = sortMode === "load" ? "" : "none";
      sortStatus.textContent = "";

      if (sortMode === "text") {
        onTextSortInput();
      } else if (sortMode === "learned") {
        updateLearnedSortDesc();
        fetchLearnedSort(true);
      }
    });
  });

  // ---- Select mode switching ----

  document.querySelectorAll('input[name="select-mode"]').forEach(radio => {
    radio.addEventListener("change", async () => {
      if (!radio.checked) return;
      selectMode = radio.value;
      if (selectMode === "new") {
        const data = await fetchDiversityTreeNext();
        if (data.id != null) {
          selectMedia(data.id);
        } else if (data.exhausted) {
          vtAlert("You have seen every branch of the diversity tree. Switch to Top or Hard mode, or add more data.", "warning");
        }
      } else {
        const nextClip = findNextClip();
        if (nextClip) selectMedia(nextClip.id);
      }
    });
  });

  // ---- Inclusion slider ----

  async function updateInclusion(newInclusion) {
    inclusion = newInclusion;
    inclusionValue.textContent = inclusion;

    // Save to server
    await fetch("/api/inclusion", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ inclusion }),
    });

    // Re-calculate threshold if in learned sort mode
    if (sortMode === "learned" && votes.good.length > 0 && votes.bad.length > 0) {
      await fetchLearnedSort();
    }
  }

  inclusionSlider.addEventListener("input", () => {
    const val = parseInt(inclusionSlider.value);
    inclusionSlider.setAttribute("aria-valuetext", String(val));
    updateInclusion(val);
  });

  // ---- Label sort dropdown ----

  const labelSortSelect = document.getElementById("label-sort-select");
  if (labelSortSelect) {
    labelSortSelect.addEventListener("change", () => {
      labelSortMode = labelSortSelect.value;
      renderVotes();
    });
  }



  // ---- Calibrate Count ----

  if (calibrateCountInput) {
    calibrateCountInput.addEventListener("change", () => {
      const val = Math.max(1, Math.min(100, parseInt(calibrateCountInput.value) || 2));
      calibrateCountInput.value = val;
      fetch("/api/settings", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ calibrate_count: val }),
      }).catch(() => {});
    });
  }

  // ---- Calibration Fraction ----

  if (calibrationFractionInput) {
    calibrationFractionInput.addEventListener("change", () => {
      const val = Math.max(0, Math.min(1, parseFloat(calibrationFractionInput.value) || 0.5));
      calibrationFractionInput.value = val;
      fetch("/api/settings", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ calibration_fraction: val }),
      }).catch(() => {});
    });
  }

  // ---- Autopilot settings ----

  if (autopilotTopGreensInput) {
    autopilotTopGreensInput.addEventListener("change", () => {
      const val = Math.max(1, parseInt(autopilotTopGreensInput.value) || 10);
      autopilotTopGreensInput.value = val;
      fetch("/api/settings", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ autopilot_top_greens: val }),
      }).catch(() => {});
    });
  }

  if (autopilotHardRedsInput) {
    autopilotHardRedsInput.addEventListener("change", () => {
      const val = Math.max(1, parseInt(autopilotHardRedsInput.value) || 10);
      autopilotHardRedsInput.value = val;
      fetch("/api/settings", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ autopilot_hard_reds: val }),
      }).catch(() => {});
    });
  }

  // ---- Text sort ----

  async function fetchTextSort(text) {
    showSortProgressWithPolling("Searching and sorting\u2026");
    try {
      const res = await fetch("/api/sort", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      if (!res.ok) {
        throw new Error(`HTTP error! status: ${res.status}`);
      }
      const data = await res.json();
      sortOrder = data.results.map(e => ({ id: e.id, score: e.similarity }));
      threshold = data.threshold;
      hideSortProgress();
      sortStatus.textContent = `Threshold: ${(threshold * 100).toFixed(1)}%`;
      renderMediaList();
      const nextClip = findNextClip();
      if (nextClip) selectMedia(nextClip.id);
    } catch (error) {
      hideSortProgress();
      sortStatus.textContent = `Error: ${error.message}`;
      console.error("Sort error:", error);
    }
  }

  function onTextSortInput() {
    clearTimeout(sortTimer);
    const text = textSortInput.value.trim();
    if (!text) {
      sortOrder = null;
      sortStatus.textContent = "";
      renderMediaList();
      return;
    }
    sortTimer = setTimeout(() => fetchTextSort(text), 400);
  }

  textSortInput.addEventListener("input", onTextSortInput);
  textSortInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      textSortInput.blur();
    }
  });

  // ---- Learned sort ----

  async function fetchLearnedSort(autoSelect = false) {
    showSortProgress("Training\u2026");
    try {
      const res = await fetch("/api/learned-sort", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      });
      if (!res.ok) {
        sortOrder = null;
        threshold = null;
        hideSortProgress();
        sortStatus.textContent = "Vote good & bad first";
        renderMediaList();
        return;
      }
      const data = await res.json();
      sortOrder = data.results;  // [{id, score}, ...]
      threshold = data.threshold;
      // Update learned_scores locally from sort results.
      const fgScores = {};
      data.results.forEach(r => { fgScores[String(r.id)] = r.score; });
      votes.learned_scores = fgScores;
      hideSortProgress();
      sortStatus.textContent = `Threshold: ${(threshold * 100).toFixed(1)}%`;
      renderMediaList();
      renderVotes();
      if (autoSelect) {
        const nextClip = findNextClip();
        if (nextClip) selectMedia(nextClip.id);
      }
    } catch (error) {
      hideSortProgress();
      sortStatus.textContent = `Error: ${error.message}`;
      console.error("Learned sort error:", error);
    }
  }

  // ---- Background learned sort (non-blocking, for use after votes) ----

  function scheduleLearnedSort(delay = 300) {
    // Debounce: reset the timer on each call so rapid votes batch into one request
    clearTimeout(learnedSortDebounce);
    learnedSortDebounce = setTimeout(() => {
      // Abort any in-flight background training request
      if (learnedSortController) {
        learnedSortController.abort();
      }
      learnedSortController = new AbortController();
      const controller = learnedSortController;

      showSortProgress("Training\u2026");

      fetch("/api/learned-sort", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: controller.signal,
      })
        .then(res => {
          if (!res.ok) {
            sortOrder = null;
            threshold = null;
            hideSortProgress();
            sortStatus.textContent = "Vote good & bad first";
            renderMediaList();
            return null;
          }
          return res.json();
        })
        .then(data => {
          if (!data) return;
          sortOrder = data.results;
          threshold = data.threshold;
          // Update learned_scores locally from sort results so confidence
          // renders immediately without an extra /api/votes round-trip.
          const newScores = {};
          data.results.forEach(r => { newScores[String(r.id)] = r.score; });
          votes.learned_scores = newScores;
          hideSortProgress();
          sortStatus.textContent = `Threshold: ${(threshold * 100).toFixed(1)}%`;
          renderMediaList();
          renderVotes();
        })
        .catch(err => {
          if (err.name === "AbortError") return; // Superseded by a newer request
          hideSortProgress();
          sortStatus.textContent = `Error: ${err.message}`;
          console.error("Learned sort error:", err);
        });
    }, delay);
  }

  // ---- Load detector sort ----

  async function fetchLoadedSort(autoSelect = false) {
    if (!loadedDetector) {
      sortStatus.textContent = "Load a sort first";
      return;
    }
    // Example-based loads already have sortOrder set — no need to re-score
    if (loadedDetector._example) {
      if (sortOrder && threshold != null) {
        sortStatus.textContent = `Threshold: ${(threshold * 100).toFixed(1)}%`;
        renderMediaList();
        if (autoSelect) {
          const nextClip = findNextClip();
          if (nextClip) selectMedia(nextClip.id);
        }
      }
      return;
    }
    showSortProgress("Scoring with loaded detector\u2026");
    try {
      const res = await fetch("/api/detector-sort", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ detector: loadedDetector }),
      });
      if (!res.ok) {
        sortOrder = null;
        threshold = null;
        hideSortProgress();
        sortStatus.textContent = "Failed to score with detector";
        renderMediaList();
        return;
      }
      const data = await res.json();
      sortOrder = data.results;  // [{id, score}, ...]
      threshold = data.threshold;
      hideSortProgress();
      sortStatus.textContent = `Threshold: ${(threshold * 100).toFixed(1)}%`;
      renderMediaList();
      if (autoSelect) {
        const nextClip = findNextClip();
        if (nextClip) selectMedia(nextClip.id);
      }
    } catch (error) {
      hideSortProgress();
      sortStatus.textContent = `Error: ${error.message}`;
      console.error("Detector sort error:", error);
    }
  }

  // ---- Load detector file ----

  // ---- Next Clip Selection ----

  async function fetchDiversityTreeNext() {
    try {
      // Send current sort scores and threshold so the diversity tree picks
      // the most surprising element within the next unseen node: the lowest-
      // scored element in above-threshold nodes, the highest in below.
      const body = {};
      if (sortOrder && sortOrder.length > 0) {
        const scores = {};
        for (const entry of sortOrder) {
          scores[entry.id] = entry.score ?? entry.similarity ?? 0;
        }
        body.scores = scores;
      }
      if (threshold !== null) {
        body.threshold = threshold;
      }
      const res = await fetch("/api/diversity-tree/next", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const data = await res.json();
      return data;  // { id: int|null, diversity_level: int, exhausted: bool }
    } catch {
      return { id: null, diversity_level: -1, exhausted: false };
    }
  }

  function findNextClip() {
    // "New" mode is handled asynchronously by callers via fetchDiversityTreeNext.
    if (selectMode === "new") return null;

    // Determine the ordered list to walk and effective threshold
    let ordered = sortOrder;
    let effectiveThreshold = threshold;
    if (!ordered || ordered.length === 0) {
      // Null sort: use medias in their current (arbitrary) order
      ordered = medias.map(c => ({ id: c.id, score: 0 }));
      // Treat threshold as the bottom for Hard mode
      effectiveThreshold = -Infinity;
    }

    if (ordered.length === 0) {
      return null;
    }

    // Get unlabeled medias (not voted on)
    const unlabeled = ordered.filter(item => {
      return !goodVoteSet.has(item.id) && !badVoteSet.has(item.id);
    });

    if (unlabeled.length === 0) {
      return null;
    }

    let nextClip;
    if (selectMode === "top") {
      // Select highest scoring unlabeled media (or first in order for null sort)
      nextClip = unlabeled[0];
    } else {
      // Select unlabeled media closest to threshold by list position,
      // breaking ties by score distance
      if (effectiveThreshold === null) {
        return null;
      }
      // Find threshold index: first position where score drops below threshold
      let thresholdIdx = ordered.length;
      for (let i = 0; i < ordered.length; i++) {
        if (ordered[i].score < effectiveThreshold) {
          thresholdIdx = i;
          break;
        }
      }
      // Map media id to its ordered index
      const idToIdx = {};
      ordered.forEach((item, idx) => { idToIdx[item.id] = idx; });

      let minIdxDist = Infinity;
      let minDist = Infinity;
      for (const item of unlabeled) {
        const idxDist = Math.abs(idToIdx[item.id] - thresholdIdx);
        const dist = Math.abs(item.score - effectiveThreshold);
        if (idxDist < minIdxDist || (idxDist === minIdxDist && dist < minDist)) {
          minIdxDist = idxDist;
          minDist = dist;
          nextClip = item;
        }
      }
    }

    return nextClip;
  }

  // ---- Rendering ----

  async function fetchMedias() {
    const res = await fetch("/api/medias");
    if (!res.ok) { console.error("fetchMedias failed:", res.status); return; }
    medias = await res.json();
    renderMediaList();
  }

  async function fetchVotes() {
    const res = await fetch("/api/votes");
    if (!res.ok) { console.error("fetchVotes failed:", res.status); return; }
    votes = await res.json();
    _rebuildVoteSets();
    renderVotes();
    renderStripe();
    updateSortModeAvailability();
    if (selected) renderCenter();
  }

  async function fetchInclusion() {
    const res = await fetch("/api/inclusion");
    if (!res.ok) { console.error("fetchInclusion failed:", res.status); return; }
    const data = await res.json();
    inclusion = data.inclusion;
    inclusionSlider.value = inclusion;
    inclusionValue.textContent = inclusion;
  }

  function mediaSupportsThumbnail(media) {
    return media && (media.type === "image" || media.type === "video");
  }

  function thumbnailUrl(media) {
    if (media.type === "image") return `/api/medias/${media.id}/image`;
    if (media.type === "video") return `/api/medias/${media.id}/video`;
    return "";
  }

  function renderMediaList() {
    mediaList.innerHTML = "";
    const scoreMap = {};
    if (sortOrder) {
      sortOrder.forEach(s => { scoreMap[s.id] = s.score; });
    }

    let ordered;
    if (sortOrder) {
      ordered = sortOrder.map(s => medias.find(c => c.id === s.id)).filter(Boolean);
    } else {
      ordered = medias;
    }

    let thresholdLineInserted = false;
    ordered.forEach(c => {
      // Insert threshold line before first media whose score falls below threshold
      if (threshold !== null && !thresholdLineInserted && scoreMap[c.id] !== undefined && scoreMap[c.id] < threshold) {
        const line = document.createElement("div");
        line.className = "media-threshold-line";
        mediaList.appendChild(line);
        thresholdLineInserted = true;
      }

      const div = document.createElement("div");
      const isGood = goodVoteSet.has(c.id);
      const isBad = badVoteSet.has(c.id);
      let className = "media-item";
      if (selected === c.id) className += " active";
      if (isGood) className += " labeled-good";
      if (isBad) className += " labeled-bad";
      div.className = className;
      div.setAttribute("role", "option");
      div.setAttribute("tabindex", "0");
      div.setAttribute("aria-selected", selected === c.id ? "true" : "false");
      const mediaLabel = c.filename || 'Media #' + c.id;
      const labelParts = [mediaLabel];
      if (isGood) labelParts.push("labeled good");
      if (isBad) labelParts.push("labeled bad");
      if (scoreMap[c.id] !== undefined) labelParts.push(`score ${(scoreMap[c.id] * 100).toFixed(1)}%`);
      div.setAttribute("aria-label", labelParts.join(", "));
      let html = "";
      const useThumbnail = showThumbnailsLeft && mediaSupportsThumbnail(c);
      if (useThumbnail) {
        div.className += " media-item-thumb";
        const poster = c.type === "video" ? ` poster="/api/medias/${c.id}/image"` : "";
        if (c.type === "video") {
          html += `<video class="media-thumbnail" src="${thumbnailUrl(c)}" muted preload="metadata"${poster}></video>`;
        } else {
          html += `<img class="media-thumbnail" src="${thumbnailUrl(c)}" alt="${escapeHtml(mediaLabel)}" loading="lazy">`;
        }
      } else {
        html += `<div style="font-weight: 500;">${escapeHtml(mediaLabel)}</div>`;
      }
      div.innerHTML = html;
      div.onclick = () => selectMedia(c.id);
      div.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") { e.preventDefault(); selectMedia(c.id); }
      });
      mediaList.appendChild(div);
    });

    renderStripe();
  }

  function selectMedia(id) {
    selected = id;
    renderMediaList();
    renderCenter();

    const activeItem = mediaList.querySelector(".media-item.active");
    if (activeItem) {
      activeItem.scrollIntoView({ block: "nearest", behavior: "smooth" });
    }

    const c = medias.find(x => x.id === id);
    if (c) {
      announce(`Selected ${c.filename || 'Media #' + c.id}`);
    }
  }

  // Stored references for window-level mouse handlers used by image pan,
  // so they can be removed before re-adding to avoid listener leaks.
  let _ivcWindowMoveHandler = null;
  let _ivcWindowUpHandler = null;

  function renderCenter() {
    const c = medias.find(x => x.id === selected);
    if (!c) return;
    const isGood = goodVoteSet.has(c.id);
    const isBad = badVoteSet.has(c.id);
    center.className = "panel-center";

    const mediaType = c.type || "audio";
    const mtInfo = mediaTypesMap[mediaType];

    // Render media player based on media type.
    // Known types get specialised players; new/unknown types fall back to
    // the generic /api/medias/{id}/media endpoint.
    let playerHTML = '';
    if (mediaType === "video") {
      playerHTML = `<video controls loop autoplay src="/api/medias/${c.id}/video" id="media-video" aria-label="${escapeHtml(c.filename || 'Video media')}" class="media-player-video"></video>`;
    } else if (mediaType === "image") {
      playerHTML = `<div class="media-player-image-wrap"><img src="/api/medias/${c.id}/image" id="media-image" alt="${escapeHtml(c.filename || 'Image media')}" class="media-player-image"></div>`;
    } else if (mediaType === "paragraph") {
      playerHTML = `
        <div id="media-paragraph" class="media-player-text">
          Loading...
        </div>`;
    } else if (mediaType === "audio") {
      // Audio/Sound – only the waveform canvas goes inside the swipe wrapper;
      // the <audio> player is placed outside so it stays fixed during the animation.
      playerHTML = `
        <canvas id="waveform-canvas" width="600" height="120" role="img" aria-label="Audio waveform visualization"></canvas>`;
    } else {
      // Unknown/new media type: try to render via generic endpoint.
      // If it loops, use a video element; otherwise use a generic embed.
      const loops = mtInfo && mtInfo.loops;
      if (loops) {
        playerHTML = `<video controls loop autoplay src="/api/medias/${c.id}/media" id="media-video" class="media-player-video"></video>`;
      } else {
        playerHTML = `<div class="media-player-image-wrap"><object data="/api/medias/${c.id}/media" class="media-player-embed">${escapeHtml(c.filename || 'Media')}</object></div>`;
      }
    }

    center.innerHTML = `
      <div class="media-swipe-wrapper" id="media-swipe-wrapper">
        ${playerHTML}
      </div>
      ${mediaType === "audio" ? `<audio controls controlslist="nodownload" loop autoplay src="/api/medias/${c.id}/audio" id="media-audio" aria-label="${escapeHtml(c.filename || 'Audio media')}"></audio>` : ''}
      ${mediaType === "image" ? `
      <div class="image-view-controls" id="image-view-controls">
        <button class="ivc-btn" id="ivc-rotate-left" title="Rotate left" aria-label="Rotate image left">&#x21BA;</button>
        <button class="ivc-btn" id="ivc-rotate-right" title="Rotate right" aria-label="Rotate image right">&#x21BB;</button>
        <label for="ivc-zoom" class="sr-only">Zoom</label>
        <input type="range" id="ivc-zoom" class="ivc-zoom-slider" min="0.25" max="5" step="0.05" value="1" title="Zoom" aria-label="Zoom level">
        <span class="ivc-zoom-label" id="ivc-zoom-label">1×</span>
        <button class="ivc-btn" id="ivc-reset" title="Reset view" aria-label="Reset image view">Reset</button>
      </div>` : ''}
      <div class="metadata-grid">
        <div class="metadata-item">
          <span class="metadata-label">Name</span>
          <span class="metadata-value">${escapeHtml(c.filename || 'Media #' + c.id)}</span>
        </div>
        ${c.frequency ? `
        <div class="metadata-item">
          <span class="metadata-label">Frequency</span>
          <span class="metadata-value">${escapeHtml(String(c.frequency))} Hz</span>
        </div>` : ''}
        ${c.category && c.category !== 'unknown' ? `
        <div class="metadata-item">
          <span class="metadata-label">Category</span>
          <span class="metadata-value">${escapeHtml(c.category)}</span>
        </div>` : ''}
        <div class="metadata-item">
          <span class="metadata-label">Media Type</span>
          <span class="metadata-value">${escapeHtml(mtInfo ? mtInfo.name : mediaType)}</span>
        </div>
        ${(c.duration && c.duration > 0) ? `
        <div class="metadata-item">
          <span class="metadata-label">Duration</span>
          <span class="metadata-value">${c.duration.toFixed(1)}s</span>
        </div>` : ''}
        ${(c.width && c.height) ? `
        <div class="metadata-item">
          <span class="metadata-label">Dimensions</span>
          <span class="metadata-value">${c.width}×${c.height}</span>
        </div>` : ''}
        ${(c.word_count) ? `
        <div class="metadata-item">
          <span class="metadata-label">Word Count</span>
          <span class="metadata-value">${c.word_count}</span>
        </div>
        <div class="metadata-item">
          <span class="metadata-label">Characters</span>
          <span class="metadata-value">${c.character_count}</span>
        </div>` : ''}
        ${c.file_size ? `<div class="metadata-item">
          <span class="metadata-label">File Size</span>
          <span class="metadata-value">${(c.file_size / 1024).toFixed(1)} KB</span>
        </div>` : ''}
        <div class="metadata-item">
          <span class="metadata-label">MD5</span>
          <span class="metadata-value metadata-md5">${escapeHtml(c.md5)}</span>
        </div>
      </div>
      <div class="vote-buttons">
        <button class="btn-bad${isBad ? " voted" : ""}" id="vote-bad" title="Mark this media as a Bad example.">Bad</button>
        <button class="btn-good${isGood ? " voted" : ""}" id="vote-good" title="Mark this media as a Good example.">Good</button>
      </div>`;
    document.getElementById("vote-good").onclick = () => castVote(c.id, "good");
    document.getElementById("vote-bad").onclick = () => castVote(c.id, "bad");

    // Draw waveform only for audio medias
    if (mediaType === "audio") {
      drawWaveform(c.id);
      const audioEl = document.getElementById("media-audio");
      if (audioEl) {
        audioEl.volume = audioVolume;
        audioEl.addEventListener("volumechange", () => {
          audioVolume = audioEl.volume;
          saveVolume(audioVolume);
        });
      }
    }

    // Load paragraph text content
    if (mediaType === "paragraph") {
      if (paragraphController) paragraphController.abort();
      paragraphController = new AbortController();
      const expectedId = c.id;
      fetch(`/api/medias/${c.id}/paragraph`, { signal: paragraphController.signal })
        .then(res => res.json())
        .then(data => {
          if (selected !== expectedId) return; // selection changed, discard stale response
          const paragraphDiv = document.getElementById("media-paragraph");
          if (paragraphDiv) {
            paragraphDiv.textContent = data.content;
          }
        })
        .catch(err => {
          if (err.name === "AbortError") return; // expected when selection changes
          console.error("Error loading paragraph:", err);
        });
    }

    // Image view controls: zoom, rotate, pan, reset
    if (mediaType === "image") {
      const img = document.getElementById("media-image");
      const wrap = img ? img.closest(".media-player-image-wrap") : null;
      const zoomSlider = document.getElementById("ivc-zoom");
      const zoomLabel = document.getElementById("ivc-zoom-label");
      const rotateLeftBtn = document.getElementById("ivc-rotate-left");
      const rotateRightBtn = document.getElementById("ivc-rotate-right");
      const resetBtn = document.getElementById("ivc-reset");
      if (img && zoomSlider && wrap) {
        let ivcZoom = 1, ivcRotation = 0, ivcPanX = 0, ivcPanY = 0;
        // Compute how far the image can be panned before showing empty space
        const getMaxPan = () => {
          const natW = img.naturalWidth;
          const natH = img.naturalHeight;
          if (!natW || !natH) return { x: 0, y: 0 };
          const wrapW = wrap.clientWidth;
          const wrapH = wrap.clientHeight;
          if (!wrapW || !wrapH) return { x: 0, y: 0 };
          const imgAspect = natW / natH;
          const wrapAspect = wrapW / wrapH;
          let rendW, rendH;
          if (imgAspect > wrapAspect) {
            rendW = wrapW;
            rendH = wrapW / imgAspect;
          } else {
            rendH = wrapH;
            rendW = wrapH * imgAspect;
          }
          const rot = ((ivcRotation % 360) + 360) % 360;
          const swapped = (rot === 90 || rot === 270);
          const effW = swapped ? rendH : rendW;
          const effH = swapped ? rendW : rendH;
          return {
            x: Math.max(0, (effW * ivcZoom - wrapW) / 2),
            y: Math.max(0, (effH * ivcZoom - wrapH) / 2),
          };
        };
        const applyTransform = () => {
          const max = getMaxPan();
          ivcPanX = Math.max(-max.x, Math.min(max.x, ivcPanX));
          ivcPanY = Math.max(-max.y, Math.min(max.y, ivcPanY));
          img.style.transform = `translate(${ivcPanX}px, ${ivcPanY}px) scale(${ivcZoom}) rotate(${ivcRotation}deg)`;
          zoomLabel.textContent = ivcZoom.toFixed(1) + '×';
          wrap.style.cursor = (max.x > 0 || max.y > 0) ? 'grab' : '';
        };
        const clampZoom = (val) => Math.min(parseFloat(zoomSlider.max), Math.max(parseFloat(zoomSlider.min), val));
        zoomSlider.addEventListener("input", () => {
          ivcZoom = parseFloat(zoomSlider.value);
          applyTransform();
        });
        rotateLeftBtn.addEventListener("click", () => {
          ivcRotation -= 90;
          applyTransform();
        });
        rotateRightBtn.addEventListener("click", () => {
          ivcRotation += 90;
          applyTransform();
        });
        resetBtn.addEventListener("click", () => {
          ivcZoom = 1; ivcRotation = 0; ivcPanX = 0; ivcPanY = 0;
          zoomSlider.value = 1;
          applyTransform();
        });

        // Mouse wheel zoom — zooms toward cursor position
        wrap.addEventListener("wheel", (e) => {
          e.preventDefault();
          const oldZoom = ivcZoom;
          const delta = e.deltaY > 0 ? -0.15 : 0.15;
          ivcZoom = clampZoom(ivcZoom + delta * ivcZoom);
          zoomSlider.value = ivcZoom;
          // Adjust pan so the point under the cursor stays fixed
          const rect = wrap.getBoundingClientRect();
          const cx = e.clientX - rect.left - rect.width / 2;
          const cy = e.clientY - rect.top - rect.height / 2;
          const ratio = ivcZoom / oldZoom;
          ivcPanX = cx - ratio * (cx - ivcPanX);
          ivcPanY = cy - ratio * (cy - ivcPanY);
          applyTransform();
        }, { passive: false });

        // Mouse drag panning
        let isPanning = false, panStartX = 0, panStartY = 0, panOriginX = 0, panOriginY = 0;
        wrap.addEventListener("mousedown", (e) => {
          const max = getMaxPan();
          if ((max.x <= 0 && max.y <= 0) || e.button !== 0) return;
          isPanning = true;
          panStartX = e.clientX; panStartY = e.clientY;
          panOriginX = ivcPanX; panOriginY = ivcPanY;
          wrap.style.cursor = 'grabbing';
          e.preventDefault();
        });
        // Remove previous window-level handlers before adding new ones
        if (_ivcWindowMoveHandler) window.removeEventListener("mousemove", _ivcWindowMoveHandler);
        if (_ivcWindowUpHandler) window.removeEventListener("mouseup", _ivcWindowUpHandler);
        _ivcWindowMoveHandler = (e) => {
          if (!isPanning) return;
          ivcPanX = panOriginX + (e.clientX - panStartX);
          ivcPanY = panOriginY + (e.clientY - panStartY);
          applyTransform();
          wrap.style.cursor = 'grabbing';
        };
        _ivcWindowUpHandler = () => {
          if (!isPanning) return;
          isPanning = false;
          const max = getMaxPan();
          wrap.style.cursor = (max.x > 0 || max.y > 0) ? 'grab' : '';
        };
        window.addEventListener("mousemove", _ivcWindowMoveHandler);
        window.addEventListener("mouseup", _ivcWindowUpHandler);
      }
    }
  }

  async function castVote(id, vote) {
    if (isVoting) return; // Prevent double-click from toggling the vote off
    isVoting = true;
    try {
      // Flash the clicked button immediately for tactile feedback.
      // If the button already has .voted (toggling off), remove it instead.
      const btnId = vote === "good" ? "vote-good" : "vote-bad";
      const clickedBtn = document.getElementById(btnId);
      const wasVoted = clickedBtn && clickedBtn.classList.contains("voted");
      if (clickedBtn) {
        if (wasVoted) {
          clickedBtn.classList.remove("voted");
        } else {
          clickedBtn.classList.add("vote-flash");
        }
      }

      const mediaName = (medias.find(c => c.id === id) || {}).filename || `Clip #${id}`;
      const voteRes = await fetch(`/api/medias/${id}/vote`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ vote }),
      });
      if (!voteRes.ok) { console.error("castVote failed:", voteRes.status); return; }
      announce(`Voted ${vote} on ${mediaName}`);
      await fetchVotes();

      // In dashboard train mode, persist labels to the trainable model's
      // labelset after every vote so work is never lost.
      if (_dashboardTrainMode) {
        _persistTrainableModelLabels(); // fire-and-forget
      }

      // When voting Good while a text-sort query is active, store the query
      // as a suggested name for saving detectors / labelsets later.
      if (vote === "good" && sortMode === "text") {
        const textQuery = textSortInput.value.trim();
        if (textQuery) {
          fetch("/api/textsort-suggestions", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text: textQuery }),
          }).catch(() => {}); // fire-and-forget
        }
      }

      // Autopilot: count hard labels and check if the phase should advance.
      _autopilotCountHardLabel();
      checkAutopilotPhase();

      // Auto-advance to next media.  When swipe animation is enabled, play a
      // fast swipe-out before switching; otherwise advance immediately.
      // In "new" select mode, use the diversity tree for the next media.
      let nextId;
      if (selectMode === "new") {
        const data = await fetchDiversityTreeNext();
        nextId = data.id;
        if (nextId == null && data.exhausted) {
          vtAlert("You have seen every branch of the diversity tree. Switch to Top or Hard mode, or add more data.", "warning");
        }
      } else {
        const c = findNextClip();
        nextId = c ? c.id : null;
      }
      if (nextId != null && nextId !== selected) {
        if (swipeAnimation) {
          const dir = vote === "good" ? "swipe-right" : "swipe-left";
          const wrapper = document.getElementById("media-swipe-wrapper");
          if (wrapper) {
            wrapper.classList.add(dir);
            await new Promise(r => setTimeout(r, 180));
            wrapper.classList.remove(dir);
          }
        }
        selectMedia(nextId);
      } else {
        // No auto-advance (all medias voted, or toggled a vote off) —
        // refresh media list and center panel so vote button state updates.
        renderMediaList();
        renderCenter();
      }

      // Kick off learned sort in the background (non-blocking).
      // When training completes, sortOrder/threshold update and the media list
      // re-renders — but the user can keep voting in the meantime.
      if (sortMode === "learned") {
        scheduleLearnedSort();
      }
    } finally {
      isVoting = false;
    }
  }

  async function drawWaveform(mediaId) {
    const canvas = document.getElementById("waveform-canvas");
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    // Size the canvas bitmap to match its CSS-displayed size for sharp rendering
    const rect = canvas.getBoundingClientRect();
    if (rect.width > 0) canvas.width = Math.round(rect.width);
    const width = canvas.width;
    const height = canvas.height;

    // Clear canvas
    ctx.fillStyle = themeColor("--bg-surface");
    ctx.fillRect(0, 0, width, height);

    try {
      // Fetch audio data
      const response = await fetch(`/api/medias/${mediaId}/audio`);
      const arrayBuffer = await response.arrayBuffer();

      // Decode audio data (reuse a single AudioContext to avoid leaks)
      if (!waveformAudioCtx || waveformAudioCtx.state === "closed") {
        waveformAudioCtx = new (window.AudioContext || window.webkitAudioContext)();
      }
      const audioBuffer = await waveformAudioCtx.decodeAudioData(arrayBuffer);

      // Get audio data from first channel
      const channelData = audioBuffer.getChannelData(0);
      const step = Math.ceil(channelData.length / width);
      const amp = height / 2;

      // Draw waveform
      ctx.strokeStyle = themeColor("--accent");
      ctx.lineWidth = 1;
      ctx.beginPath();

      for (let i = 0; i < width; i++) {
        let min = 1.0;
        let max = -1.0;

        for (let j = 0; j < step; j++) {
          const datum = channelData[i * step + j];
          if (datum < min) min = datum;
          if (datum > max) max = datum;
        }

        const yMin = (1 + min) * amp;
        const yMax = (1 + max) * amp;

        if (i === 0) {
          ctx.moveTo(i, yMin);
        }
        ctx.lineTo(i, yMin);
        ctx.lineTo(i, yMax);
      }

      ctx.stroke();

      // Draw center line
      ctx.strokeStyle = themeColor("--border");
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, height / 2);
      ctx.lineTo(width, height / 2);
      ctx.stroke();

    } catch (error) {
      console.error("Error drawing waveform:", error);
      // Draw error message
      ctx.fillStyle = themeColor("--color-bad");
      ctx.font = "12px monospace";
      ctx.textAlign = "center";
      ctx.fillText("Unable to load waveform", width / 2, height / 2);
    }
  }

  function labelSortKey(id, label) {
    const media = medias.find(c => c.id === id);
    const name = media ? (media.filename || `Clip #${id}`) : `Clip #${id}`;
    const time = votes.click_times[String(id)] ?? -1;
    const score = votes.learned_scores[String(id)] ?? -1;
    // Confidence: for "good" labels, higher score = more confident.
    // For "bad" labels, lower score = more confident (so use 1 - score).
    let confidence = -1;
    if (score >= 0) {
      confidence = label === "good" ? score : 1 - score;
    }
    return { id, name, time, score, confidence };
  }

  function sortLabelEntries(ids, label) {
    const entries = ids.map(id => labelSortKey(id, label));
    switch (labelSortMode) {
      case "time-desc":
        entries.sort((a, b) => b.time - a.time);
        break;
      case "time-asc":
        entries.sort((a, b) => a.time - b.time);
        break;
      case "name-asc":
        entries.sort((a, b) => a.name.localeCompare(b.name));
        break;
      case "name-desc":
        entries.sort((a, b) => b.name.localeCompare(a.name));
        break;
      case "confidence-desc":
        entries.sort((a, b) => b.confidence - a.confidence);
        break;
      case "confidence-asc":
        entries.sort((a, b) => a.confidence - b.confidence);
        break;
      case "id-asc":
      default:
        entries.sort((a, b) => a.id - b.id);
        break;
    }
    return entries;
  }

  function renderVoteEntry(entry, label, parentEl) {
    const media = medias.find(c => c.id === entry.id);
    const div = document.createElement("div");
    div.className = "vote-entry";
    div.setAttribute("role", "button");
    div.setAttribute("tabindex", "0");
    div.setAttribute("aria-label", `${label}: ${entry.name}`);
    const metaParts = [];
    if (entry.time >= 0) metaParts.push(`#${entry.time}`);
    else metaParts.push("imported");
    if (entry.confidence >= 0) metaParts.push(`${(entry.confidence * 100).toFixed(0)}%`);

    const useThumbnail = showThumbnailsRight && mediaSupportsThumbnail(media);
    let html = "";
    if (useThumbnail) {
      div.className += " vote-entry-thumb";
      if (media.type === "video") {
        html += `<video class="vote-thumbnail" src="${thumbnailUrl(media)}" muted preload="metadata"></video>`;
      } else {
        html += `<img class="vote-thumbnail" src="${thumbnailUrl(media)}" alt="${escapeHtml(entry.name)}" loading="lazy">`;
      }
      html += `<div class="vote-thumb-info">`;
    }
    html += `<span class="vote-name">${escapeHtml(entry.name)}</span><span class="vote-meta">${metaParts.join(" \u00b7 ")}</span>`;
    if (useThumbnail) {
      html += `</div>`;
    }
    div.innerHTML = html;
    div.onclick = () => selectMedia(entry.id);
    div.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); selectMedia(entry.id); }
    });
    parentEl.appendChild(div);
  }

  function renderVotes() {
    goodList.innerHTML = "";
    badList.innerHTML = "";

    const sortedGood = sortLabelEntries(votes.good, "good");
    sortedGood.forEach(entry => renderVoteEntry(entry, "Good", goodList));

    const sortedBad = sortLabelEntries(votes.bad, "bad");
    sortedBad.forEach(entry => renderVoteEntry(entry, "Bad", badList));
  }

  function renderStripe() {
    stripeContainer.innerHTML = "";

    // Only show stripe when sorted
    if (!sortOrder || sortOrder.length === 0) {
      stripeOverview.style.display = "none";
      return;
    }

    stripeOverview.style.display = "block";
    const totalClips = sortOrder.length;

    // Add highlight element
    const highlight = document.createElement("div");
    highlight.className = "stripe-highlight";
    stripeContainer.appendChild(highlight);

    // Render dots for each labeled media
    sortOrder.forEach((item, index) => {
      const isGood = goodVoteSet.has(item.id);
      const isBad = badVoteSet.has(item.id);
      const isSelected = item.id === selected;

      if (isGood || isBad) {
        const dot = document.createElement("div");
        dot.className = `stripe-dot ${isGood ? "good" : "bad"}`;
        dot.style.top = `${(index / totalClips) * 100}%`;
        dot.setAttribute("data-media-id", item.id);
        dot.setAttribute("data-index", index);
        stripeContainer.appendChild(dot);
      }

      if (isSelected) {
        const dot = document.createElement("div");
        dot.className = "stripe-dot selected";
        dot.style.top = `${(index / totalClips) * 100}%`;
        stripeContainer.appendChild(dot);
      }
    });

    // Render threshold line if available
    if (threshold !== null) {
      // Find the index where score crosses threshold
      let thresholdIndex = sortOrder.length;
      for (let i = 0; i < sortOrder.length; i++) {
        if (sortOrder[i].score < threshold) {
          thresholdIndex = i;
          break;
        }
      }

      const thresholdLine = document.createElement("div");
      thresholdLine.className = "stripe-threshold";
      thresholdLine.style.top = `${(thresholdIndex / totalClips) * 100}%`;
      stripeContainer.appendChild(thresholdLine);
    }

    updateStripeHighlight();
  }

  // ---- Stripe click handler ----

  stripeOverview.addEventListener("click", (e) => {
    if (!sortOrder || sortOrder.length === 0) return;

    const rect = stripeOverview.getBoundingClientRect();
    const y = e.clientY - rect.top;
    const percentage = y / rect.height;
    const index = Math.floor(percentage * sortOrder.length);
    const clampedIndex = Math.max(0, Math.min(index, sortOrder.length - 1));

    if (sortOrder[clampedIndex]) {
      const mediaId = sortOrder[clampedIndex].id;
      selectMedia(mediaId);

      // Scroll the media into view
      const mediaItems = mediaList.querySelectorAll(".media-item");
      if (mediaItems[clampedIndex]) {
        mediaItems[clampedIndex].scrollIntoView({ behavior: "smooth", block: "center" });
      }
    }
  });

  // ---- Stripe highlight update ----

  function updateStripeHighlight() {
    const highlight = stripeContainer.querySelector(".stripe-highlight");
    if (!highlight) return;

    const scrollHeight = mediaList.scrollHeight;
    const clientHeight = mediaList.clientHeight;
    const scrollTop = mediaList.scrollTop;

    const topPercent = scrollHeight > 0 ? (scrollTop / scrollHeight) * 100 : 0;
    const heightPercent = scrollHeight > 0 ? (clientHeight / scrollHeight) * 100 : 100;

    highlight.style.top = `${topPercent}%`;
    highlight.style.height = `${heightPercent}%`;
  }

  mediaList.addEventListener("scroll", updateStripeHighlight);
  window.addEventListener("resize", updateStripeHighlight);

  // ---- Label importer modal ----

  async function openLabelImporterModal() {
    // Fetch available importers
    let importers = [];
    try {
      const res = await fetch("/api/label-importers");
      if (res.ok) importers = await res.json();
    } catch (_) { /* ignore */ }

    // Reset to picker view
    labelImporterFormDiv.style.display = "none";
    labelImporterFormDiv.innerHTML = "";
    labelImporterBack.style.display = "none";
    labelImporterList.style.display = "";

    if (importers.length === 0) {
      labelImporterList.innerHTML = '<p style="color:var(--text-muted);">No label importers available.</p>';
    } else {
      labelImporterList.innerHTML = importers.map(imp => `
        <div class="label-importer-option option-card" data-name="${escapeHtml(imp.name)}">
          <span class="option-card-icon">${escapeHtml(imp.icon || '🏷️')}</span>
          <div>
            <div class="option-card-title">${escapeHtml(imp.display_name)}</div>
            <div class="option-card-desc">${escapeHtml(imp.description)}</div>
          </div>
        </div>
      `).join("");

      labelImporterList.querySelectorAll(".label-importer-option").forEach(el => {
        el.setAttribute("role", "button");
        el.setAttribute("tabindex", "0");
        const name = el.dataset.name;
        const imp = importers.find(i => i.name === name);
        el.addEventListener("click", () => showLabelImporterForm(imp));
        el.addEventListener("keydown", (e) => {
          if (e.key === "Enter" || e.key === " ") { e.preventDefault(); showLabelImporterForm(imp); }
        });
      });
    }

    labelImporterModal.classList.add("show");
  }

  function _showMissingElementsPrompt(statusEl, result, formEl) {
    const n = result.missing_count;
    const promptDiv = document.createElement("div");
    promptDiv.className = "missing-prompt";
    promptDiv.innerHTML = `
      <div style="color:var(--text-primary);margin-bottom:10px;font-size:0.9rem;">
        <strong>${n}</strong> element(s) from the labelset were not found in your dataset.
        Import them from their origins?
      </div>
      <div style="display:flex;gap:10px;">
        <button id="missing-import-btn" class="btn-block-primary" style="flex:1;width:auto;">Import medias</button>
        <button id="missing-skip-btn" class="btn-secondary-block">Skip</button>
      </div>
      <div id="missing-status" class="status-text" style="margin-top:8px;"></div>
    `;
    // Insert after the status element
    statusEl.parentNode.appendChild(promptDiv);

    promptDiv.querySelector("#missing-skip-btn").addEventListener("click", () => {
      promptDiv.remove();
      setTimeout(() => {
        labelImporterModal.classList.remove("show");
        menuLabelsStatus.textContent = `Applied ${result.applied} label(s)`;
        setTimeout(() => { menuLabelsStatus.textContent = ""; }, 3000);
      }, 300);
    });

    promptDiv.querySelector("#missing-import-btn").addEventListener("click", async () => {
      const missingStatus = promptDiv.querySelector("#missing-status");
      const importBtn = promptDiv.querySelector("#missing-import-btn");
      importBtn.disabled = true;
      importBtn.style.opacity = "0.5";
      missingStatus.textContent = "Ingesting medias from origins\u2026";
      missingStatus.style.color = "var(--text-muted)";

      try {
        const ingestRes = await fetch("/api/label-importers/ingest-missing", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({entries: result.missing}),
        });
        const ingestResult = await ingestRes.json();
        if (ingestRes.ok) {
          missingStatus.textContent = ingestResult.message;
          missingStatus.style.color = "var(--color-good)";
          await fetchVotes();
          const totalApplied = result.applied + (ingestResult.applied || 0);
          setTimeout(() => {
            labelImporterModal.classList.remove("show");
            menuLabelsStatus.textContent = `Applied ${totalApplied} label(s)`;
            setTimeout(() => { menuLabelsStatus.textContent = ""; }, 3000);
          }, 1500);
        } else {
          missingStatus.textContent = ingestResult.error || "Ingest failed";
          missingStatus.style.color = "var(--color-bad)";
          importBtn.disabled = false;
          importBtn.style.opacity = "1";
        }
      } catch (err) {
        missingStatus.textContent = `Error: ${err.message}`;
        missingStatus.style.color = "var(--color-bad)";
        importBtn.disabled = false;
        importBtn.style.opacity = "1";
      }
    });
  }

  function showLabelImporterForm(importer) {
    labelImporterList.style.display = "none";
    labelImporterBack.style.display = "inline-block";

    let html = `<h3 class="form-heading">${escapeHtml(importer.display_name)}</h3>`;
    html += `<form id="label-imp-form">`;
    for (const field of importer.fields) {
      html += `<div class="form-group">`;
      html += `<label class="form-label">${escapeHtml(field.label)}${field.required ? " *" : ""}</label>`;
      if (field.field_type === "file") {
        html += `<input type="file" name="${escapeHtml(field.key)}" accept="${escapeHtml(field.accept)}" class="form-input" ${field.required ? "required" : ""}>`;
      } else if (field.field_type === "select") {
        html += `<select name="${escapeHtml(field.key)}" class="form-input">`;
        for (const opt of field.options) {
          html += `<option value="${escapeHtml(opt)}"${opt === field.default ? " selected" : ""}>${escapeHtml(opt)}</option>`;
        }
        html += `</select>`;
      } else {
        const itype = field.field_type === "password" ? "password" : "text";
        const placeholder = escapeHtml(field.placeholder || field.description);
        html += `<input type="${itype}" name="${escapeHtml(field.key)}" value="${escapeHtml(field.default)}" placeholder="${placeholder}" class="form-input" ${field.required ? "required" : ""}>`;
      }
      if (field.description) {
        html += `<div class="form-hint">${escapeHtml(field.description)}</div>`;
      }
      html += `</div>`;
    }
    html += `<div id="label-imp-status" class="status-text compact"></div>`;
    html += `<button type="submit" class="btn-block-primary">Import</button>`;
    html += `</form>`;

    labelImporterFormDiv.innerHTML = html;
    labelImporterFormDiv.style.display = "block";

    const statusEl = labelImporterFormDiv.querySelector("#label-imp-status");

    labelImporterFormDiv.querySelector("#label-imp-form").addEventListener("submit", async (e) => {
      e.preventDefault();
      statusEl.textContent = "Importing\u2026";
      statusEl.style.color = "var(--text-muted)";

      const formEl = e.target;
      const hasFiles = importer.fields.some(f => f.field_type === "file");
      let body, headers = {};

      if (hasFiles) {
        body = new FormData(formEl);
      } else {
        const obj = {};
        for (const field of importer.fields) {
          obj[field.key] = formEl.elements[field.key].value;
        }
        body = JSON.stringify(obj);
        headers["Content-Type"] = "application/json";
      }

      try {
        const res = await fetch(`/api/label-importers/import/${encodeURIComponent(importer.name)}`, {
          method: "POST", headers, body,
        });
        const result = await res.json();
        if (res.ok) {
          statusEl.textContent = `Applied ${result.applied}, skipped ${result.skipped}.`;
          statusEl.style.color = "var(--color-good)";
          await fetchVotes();

          if (result.missing_count > 0) {
            // Show prompt for missing elements
            _showMissingElementsPrompt(statusEl, result, formEl);
          } else {
            setTimeout(() => {
              labelImporterModal.classList.remove("show");
              menuLabelsStatus.textContent = `Applied ${result.applied} label(s)`;
              setTimeout(() => { menuLabelsStatus.textContent = ""; }, 3000);
            }, 1500);
          }
        } else {
          statusEl.textContent = result.error || "Import failed";
          statusEl.style.color = "var(--color-bad)";
        }
      } catch (err) {
        statusEl.textContent = `Error: ${err.message}`;
        statusEl.style.color = "var(--color-bad)";
      }
    });
  }

  if (labelImporterModalClose) {
    labelImporterModalClose.addEventListener("click", () => {
      labelImporterModal.classList.remove("show");
    });
  }

  if (labelExporterModalClose) {
    labelExporterModalClose.addEventListener("click", () => {
      labelExporterModal.classList.remove("show");
    });
  }

  if (detectorExportModalClose) {
    detectorExportModalClose.addEventListener("click", () => {
      detectorExportModal.classList.remove("show");
    });
  }

  if (labelImporterBack) {
    labelImporterBack.addEventListener("click", () => {
      labelImporterFormDiv.style.display = "none";
      labelImporterFormDiv.innerHTML = "";
      labelImporterBack.style.display = "none";
      labelImporterList.style.display = "";
    });
  }

  // ---- Load Sort modal ----

  if (loadSortModalClose) {
    loadSortModalClose.addEventListener("click", () => {
      loadSortModal.classList.remove("show");
      // If no detector/example was loaded, revert to text mode
      if (!loadedDetector) {
        document.querySelector('input[name="sort-mode"][value="text"]').checked = true;
        sortMode = "text";
        textSortWrap.style.display = "";
        learnedSortWrap.style.display = "none";
        loadSortWrap.style.display = "none";
      }
    });
  }

  /**
   * Activate load sort mode and show results in the left column.
   * @param {string} label - Status text describing the loaded sort source.
   */
  function activateLoadSort(label) {
    sortMode = "load";
    document.querySelector('input[name="sort-mode"][value="load"]').checked = true;
    textSortWrap.style.display = "none";
    learnedSortWrap.style.display = "none";
    loadSortWrap.style.display = "";
    loadSortDesc.textContent = label;
    updateSortModeAvailability();
  }

  async function openLoadSortModal() {
    if (!loadSortModal) return;
    loadSortStatus.textContent = "";

    // Determine current media type for the file accept filter
    let mediaType = "audio";
    let acceptExts = ".wav,.mp3,.flac,.ogg,.m4a";
    if (medias.length > 0 && medias[0].type) {
      mediaType = medias[0].type;
    }
    // Look up file extensions from the mediaTypesMap
    if (mediaTypesMap[mediaType] && mediaTypesMap[mediaType].file_extensions) {
      acceptExts = mediaTypesMap[mediaType].file_extensions
        .map(ext => ext.replace("*", ""))
        .join(",");
    }
    loadSortMediaFile.setAttribute("accept", acceptExts);

    // Build detector options
    let detectorHtml = `
      <div class="load-sort-option option-card" id="ls-detector-local" role="button" tabindex="0">
        <span class="option-card-icon">\uD83D\uDCC1</span>
        <div>
          <div class="option-card-title">Load Local Detector</div>
          <div class="option-card-desc">Choose a detector JSON file from your computer.</div>
        </div>
      </div>`;

    // Fetch server detector files
    let serverDetectors = [];
    try {
      const res = await fetch("/api/detector/server-files");
      if (res.ok) {
        const data = await res.json();
        serverDetectors = data.files || [];
      }
    } catch (_) { /* ignore */ }

    if (serverDetectors.length > 0) {
      detectorHtml += `
        <div class="load-sort-option option-card" id="ls-detector-server" role="button" tabindex="0">
          <span class="option-card-icon">\uD83D\uDCBE</span>
          <div>
            <div class="option-card-title">Load Server Detector</div>
            <div class="option-card-desc">${serverDetectors.length} detector file${serverDetectors.length !== 1 ? "s" : ""} on the server.</div>
          </div>
        </div>`;
    }

    // Fetch processor importers (label-based detector sources)
    let procImporters = [];
    try {
      const res = await fetch("/api/processor-importers");
      if (res.ok) procImporters = await res.json();
    } catch (_) { /* ignore */ }

    for (const imp of procImporters) {
      detectorHtml += `
        <div class="load-sort-option option-card" data-proc-importer="${escapeHtml(imp.name)}" role="button" tabindex="0">
          <span class="option-card-icon">${escapeHtml(imp.icon || '\u{1F9E9}')}</span>
          <div>
            <div class="option-card-title">${escapeHtml(imp.display_name)}</div>
            <div class="option-card-desc">${escapeHtml(imp.description)}</div>
          </div>
        </div>`;
    }

    loadSortDetectorOptions.innerHTML = detectorHtml;

    // Build example options
    let exampleHtml = `
      <div class="load-sort-option option-card" id="ls-example-local" role="button" tabindex="0">
        <span class="option-card-icon">\uD83D\uDCC1</span>
        <div>
          <div class="option-card-title">Local Example</div>
          <div class="option-card-desc">Choose a ${mediaType} file from your computer to sort by similarity.</div>
        </div>
      </div>`;

    // Fetch server media files
    let serverMediaFiles = [];
    try {
      const res = await fetch("/api/server-media-files");
      if (res.ok) {
        const data = await res.json();
        serverMediaFiles = data.files || [];
      }
    } catch (_) { /* ignore */ }

    const serverExampleDesc = serverMediaFiles.length > 0
      ? `${serverMediaFiles.length} media file${serverMediaFiles.length !== 1 ? "s" : ""} on the server.`
      : "No example media files saved on server yet.";
    exampleHtml += `
      <div class="load-sort-option option-card${serverMediaFiles.length === 0 ? " option-card-disabled" : ""}" id="ls-example-server" role="button" tabindex="0">
        <span class="option-card-icon">\uD83D\uDCBE</span>
        <div>
          <div class="option-card-title">Server Example</div>
          <div class="option-card-desc">${serverExampleDesc}</div>
        </div>
      </div>`;

    loadSortExampleOptions.innerHTML = exampleHtml;

    // --- Wire up click handlers ---

    // Local detector file
    const lsDetectorLocal = document.getElementById("ls-detector-local");
    if (lsDetectorLocal) {
      lsDetectorLocal.addEventListener("click", () => {
        loadSortModal.classList.remove("show");
        loadSortDetectorFile.click();
      });
    }

    // Server detector file — show sub-list
    const lsDetectorServer = document.getElementById("ls-detector-server");
    if (lsDetectorServer && serverDetectors.length > 0) {
      lsDetectorServer.addEventListener("click", () => {
        loadSortDetectorOptions.innerHTML = serverDetectors.map(f => `
          <div class="load-sort-option option-card ls-server-det-item" data-det-name="${escapeHtml(f.name)}" role="button" tabindex="0">
            <span class="option-card-icon">\uD83D\uDCC4</span>
            <div>
              <div class="option-card-title">${escapeHtml(f.name)}</div>
              <div class="option-card-desc">${(f.size_bytes / 1024).toFixed(1)} KB</div>
            </div>
          </div>
        `).join("");
        loadSortExampleOptions.style.display = "none";
        loadSortDetectorOptions.querySelectorAll(".ls-server-det-item").forEach(el => {
          el.addEventListener("click", async () => {
            const name = el.dataset.detName;
            loadSortStatus.textContent = "Loading server detector\u2026";
            try {
              const res = await fetch(`/api/detector/server-files/${encodeURIComponent(name)}`);
              if (!res.ok) throw new Error("Failed to load detector");
              loadedDetector = await res.json();
              loadSortModal.classList.remove("show");
              activateLoadSort("Detector: " + name);
              fetchLoadedSort(true);
            } catch (err) {
              loadSortStatus.textContent = `Error: ${err.message}`;
            }
          });
        });
      });
    }

    // Processor importers — open inline form for training a detector
    loadSortDetectorOptions.querySelectorAll("[data-proc-importer]").forEach(el => {
      const impName = el.dataset.procImporter;
      const imp = procImporters.find(i => i.name === impName);
      el.addEventListener("click", () => {
        // Show inline form inside the modal
        showLoadSortProcessorImporterForm(imp);
      });
    });

    // Local example media
    const lsExampleLocal = document.getElementById("ls-example-local");
    if (lsExampleLocal) {
      lsExampleLocal.addEventListener("click", () => {
        loadSortModal.classList.remove("show");
        loadSortMediaFile.click();
      });
    }

    // Server example media — show sub-list (always shown; guard empty case)
    const lsExampleServer = document.getElementById("ls-example-server");
    if (lsExampleServer) {
      lsExampleServer.addEventListener("click", () => {
        if (serverMediaFiles.length === 0) {
          loadSortStatus.textContent = "No example media files on server. Place files in data/example_media/ to use this option.";
          return;
        }
        loadSortExampleOptions.innerHTML = serverMediaFiles.map(f => `
          <div class="load-sort-option option-card ls-server-media-item" data-media-filename="${escapeHtml(f.filename)}" data-media-name="${escapeHtml(f.name)}" role="button" tabindex="0">
            <span class="option-card-icon">\uD83C\uDFB5</span>
            <div>
              <div class="option-card-title">${escapeHtml(f.name)}</div>
              <div class="option-card-desc">${(f.size_bytes / 1024).toFixed(1)} KB</div>
            </div>
          </div>
        `).join("");
        loadSortDetectorOptions.style.display = "none";
        loadSortExampleOptions.querySelectorAll(".ls-server-media-item").forEach(el => {
          el.addEventListener("click", async () => {
            const filename = el.dataset.mediaFilename;
            const name = el.dataset.mediaName;
            loadSortModal.classList.remove("show");
            activateLoadSort("Example: " + name);
            showSortProgress("Scoring with example media\u2026");
            try {
              const res = await fetch("/api/example-sort-server", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ filename }),
              });
              if (!res.ok) throw new Error("Failed to sort by example");
              const data = await res.json();
              // Normalise key name: example-sort uses "similarity", detector uses "score"
              sortOrder = data.results.map(r => ({ id: r.id, score: r.similarity ?? r.score }));
              threshold = data.threshold;
              // Mark as example-based load (no loadedDetector weights)
              loadedDetector = { _example: true, _name: name };
              hideSortProgress();
              sortStatus.textContent = `Threshold: ${(threshold * 100).toFixed(1)}%`;
              renderMediaList();
              const nextClip = findNextClip();
              if (nextClip) selectMedia(nextClip.id);
            } catch (err) {
              hideSortProgress();
              sortStatus.textContent = `Error: ${err.message}`;
            }
          });
        });
      });
    }

    // Reset visibility
    loadSortDetectorOptions.style.display = "";
    loadSortExampleOptions.style.display = "";

    loadSortModal.classList.add("show");
  }

  /**
   * Show an inline processor-importer form inside the Load Sort modal.
   * This trains a detector from a label source and loads it for sorting.
   */
  function showLoadSortProcessorImporterForm(importer) {
    // Replace detector options area with the form
    let html = `<h3 class="form-heading">${escapeHtml(importer.display_name)}</h3>`;
    html += `<form id="ls-proc-imp-form">`;
    html += `<div class="form-group">`;
    html += `<label class="form-label">Detector Name *</label>`;
    html += `<input type="text" name="name" placeholder="e.g. Dog Barks" class="form-input" required>`;
    html += `</div>`;
    for (const field of importer.fields) {
      html += `<div class="form-group">`;
      html += `<label class="form-label">${escapeHtml(field.label)}${field.required ? " *" : ""}</label>`;
      if (field.field_type === "file") {
        html += `<input type="file" name="${escapeHtml(field.key)}" accept="${escapeHtml(field.accept)}" class="form-input" ${field.required ? "required" : ""}>`;
      } else if (field.field_type === "select") {
        html += `<select name="${escapeHtml(field.key)}" class="form-input">`;
        for (const opt of field.options) {
          html += `<option value="${escapeHtml(opt)}"${opt === field.default ? " selected" : ""}>${escapeHtml(opt || "(auto-detect)")}</option>`;
        }
        html += `</select>`;
      } else {
        const itype = field.field_type === "password" ? "password" : "text";
        html += `<input type="${itype}" name="${escapeHtml(field.key)}" value="${escapeHtml(field.default || "")}" placeholder="${escapeHtml(field.placeholder || field.description)}" class="form-input" ${field.required ? "required" : ""}>`;
      }
      if (field.description) html += `<div class="form-hint">${escapeHtml(field.description)}</div>`;
      html += `</div>`;
    }
    html += `<div id="ls-proc-status" class="status-text compact"></div>`;
    html += `<div style="display:flex;gap:8px;">`;
    html += `<button type="button" id="ls-proc-back" class="btn-sm" style="padding:6px 14px;">\u2190 Back</button>`;
    html += `<button type="submit" class="btn-block-primary" style="flex:1;">Import & Sort</button>`;
    html += `</div>`;
    html += `</form>`;

    loadSortDetectorOptions.innerHTML = html;
    loadSortExampleOptions.style.display = "none";

    // Back button
    document.getElementById("ls-proc-back").addEventListener("click", () => {
      openLoadSortModal();
    });

    // Form submission
    document.getElementById("ls-proc-imp-form").addEventListener("submit", async (e) => {
      e.preventDefault();
      const statusEl = document.getElementById("ls-proc-status");
      statusEl.textContent = "Importing\u2026";
      statusEl.style.color = "var(--text-muted)";

      const formEl = e.target;
      const hasFiles = importer.fields.some(f => f.field_type === "file");
      let body, headers = {};

      if (hasFiles) {
        body = new FormData(formEl);
      } else {
        const obj = { name: formEl.elements["name"].value };
        for (const field of importer.fields) {
          obj[field.key] = formEl.elements[field.key].value;
        }
        body = JSON.stringify(obj);
        headers["Content-Type"] = "application/json";
      }

      try {
        const res = await fetch(`/api/processor-importers/import/${encodeURIComponent(importer.name)}`, {
          method: "POST", headers, body,
        });
        const result = await res.json();
        if (!res.ok) throw new Error(result.error || "Import failed");

        // Now load the newly created detector and use it for sorting
        const detRes = await fetch("/api/autorun-detectors");
        if (detRes.ok) {
          const detData = await detRes.json();
          const det = (detData.detectors || []).find(d => d.name === result.name);
          if (det) {
            loadedDetector = { weights: det.weights, threshold: det.threshold, media_type: det.media_type, name: det.name };
            loadSortModal.classList.remove("show");
            activateLoadSort("Detector: " + det.name);
            fetchLoadedSort(true);
            return;
          }
        }
        statusEl.textContent = "Imported but could not load detector.";
        statusEl.style.color = "var(--color-bad)";
      } catch (err) {
        statusEl.textContent = `Error: ${err.message}`;
        statusEl.style.color = "var(--color-bad)";
      }
    });
  }

  // Handle local detector file selection from Load Sort modal
  loadSortDetectorFile.addEventListener("change", async () => {
    const file = loadSortDetectorFile.files[0];
    loadSortDetectorFile.value = "";
    if (!file) {
      // User cancelled — revert to text if no detector loaded
      if (!loadedDetector) {
        document.querySelector('input[name="sort-mode"][value="text"]').checked = true;
        sortMode = "text";
        textSortWrap.style.display = "";
        learnedSortWrap.style.display = "none";
        loadSortWrap.style.display = "none";
      }
      return;
    }
    sortStatus.textContent = "Loading detector\u2026";
    try {
      const text = await file.text();
      loadedDetector = JSON.parse(text);
      activateLoadSort("Detector: " + (loadedDetector.name || file.name.replace(/\.json$/, "")));
      fetchLoadedSort(true);
    } catch (e) {
      sortStatus.textContent = "Invalid detector file";
      loadedDetector = null;
      updateSortModeAvailability();
      document.querySelector('input[name="sort-mode"][value="text"]').checked = true;
      sortMode = "text";
      textSortWrap.style.display = "";
      learnedSortWrap.style.display = "none";
      loadSortWrap.style.display = "none";
    }
  });

  // Handle local example media selection from Load Sort modal
  loadSortMediaFile.addEventListener("change", async () => {
    const file = loadSortMediaFile.files[0];
    loadSortMediaFile.value = "";
    if (!file) {
      if (!loadedDetector) {
        document.querySelector('input[name="sort-mode"][value="text"]').checked = true;
        sortMode = "text";
        textSortWrap.style.display = "";
        learnedSortWrap.style.display = "none";
        loadSortWrap.style.display = "none";
      }
      return;
    }
    const name = file.name.replace(/\.[^.]+$/, "");
    activateLoadSort("Example: " + name);
    showSortProgress("Scoring with example media\u2026");
    try {
      const formData = new FormData();
      formData.append("file", file);
      const res = await fetch("/api/example-sort", { method: "POST", body: formData });
      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.error || "Failed to sort by example");
      }
      const data = await res.json();
      sortOrder = data.results.map(r => ({ id: r.id, score: r.similarity ?? r.score }));
      threshold = data.threshold;
      loadedDetector = { _example: true, _name: name };
      hideSortProgress();
      sortStatus.textContent = `Threshold: ${(threshold * 100).toFixed(1)}%`;
      renderMediaList();
      const nextClip = findNextClip();
      if (nextClip) selectMedia(nextClip.id);
    } catch (err) {
      hideSortProgress();
      sortStatus.textContent = `Error: ${err.message}`;
      loadedDetector = null;
      updateSortModeAvailability();
      document.querySelector('input[name="sort-mode"][value="text"]').checked = true;
      sortMode = "text";
      textSortWrap.style.display = "";
      learnedSortWrap.style.display = "none";
      loadSortWrap.style.display = "none";
    }
  });

  // ---- Processor importer modal ----

  async function openProcessorImporterModal() {
    let importers = [];
    try {
      const res = await fetch("/api/processor-importers");
      if (res.ok) importers = await res.json();
    } catch (_) { /* ignore */ }

    processorImporterFormDiv.style.display = "none";
    processorImporterFormDiv.innerHTML = "";
    processorImporterBack.style.display = "none";
    processorImporterList.style.display = "";
    const modalTitle = document.getElementById("processor-importer-modal-title");
    if (modalTitle) modalTitle.textContent = "Import Detector";

    if (importers.length === 0) {
      processorImporterList.innerHTML = '<p style="color:var(--text-muted);">No processor importers available.</p>';
    } else {
      processorImporterList.innerHTML = importers.map(imp => `
        <div class="processor-importer-option option-card" data-name="${escapeHtml(imp.name)}">
          <span class="option-card-icon">${escapeHtml(imp.icon || '\u{1F9E9}')}</span>
          <div>
            <div class="option-card-title">${escapeHtml(imp.display_name)}</div>
            <div class="option-card-desc">${escapeHtml(imp.description)}</div>
          </div>
        </div>
      `).join("");

      processorImporterList.querySelectorAll(".processor-importer-option").forEach(el => {
        el.setAttribute("role", "button");
        el.setAttribute("tabindex", "0");
        const name = el.dataset.name;
        const imp = importers.find(i => i.name === name);
        el.addEventListener("click", () => showProcessorImporterForm(imp));
        el.addEventListener("keydown", (e) => {
          if (e.key === "Enter" || e.key === " ") { e.preventDefault(); showProcessorImporterForm(imp); }
        });
      });
    }

    processorImporterModal.classList.add("show");
  }

  function showProcessorImporterForm(importer) {
    processorImporterList.style.display = "none";
    processorImporterBack.style.display = "inline-block";

    let html = `<h3 class="form-heading">${escapeHtml(importer.display_name)}</h3>`;
    html += `<form id="proc-imp-form">`;
    // Name field (always required)
    html += `<div class="form-group">`;
    html += `<label class="form-label">Detector Name *</label>`;
    html += `<input type="text" name="name" placeholder="e.g. Dog Barks" class="form-input" required>`;
    html += `<div class="form-hint">Name for the imported detector.</div>`;
    html += `</div>`;
    for (const field of importer.fields) {
      html += `<div class="form-group">`;
      html += `<label class="form-label">${escapeHtml(field.label)}${field.required ? " *" : ""}</label>`;
      if (field.field_type === "file") {
        html += `<input type="file" name="${escapeHtml(field.key)}" accept="${escapeHtml(field.accept)}" class="form-input" ${field.required ? "required" : ""}>`;
      } else if (field.field_type === "select") {
        html += `<select name="${escapeHtml(field.key)}" class="form-input">`;
        for (const opt of field.options) {
          html += `<option value="${escapeHtml(opt)}"${opt === field.default ? " selected" : ""}>${escapeHtml(opt || "(auto-detect)")}</option>`;
        }
        html += `</select>`;
      } else {
        const itype = field.field_type === "password" ? "password" : "text";
        const placeholder = escapeHtml(field.placeholder || field.description);
        html += `<input type="${itype}" name="${escapeHtml(field.key)}" value="${escapeHtml(field.default)}" placeholder="${placeholder}" class="form-input" ${field.required ? "required" : ""}>`;
      }
      if (field.description) {
        html += `<div class="form-hint">${escapeHtml(field.description)}</div>`;
      }
      html += `</div>`;
    }
    html += `<div id="proc-imp-status" class="status-text compact"></div>`;
    html += `<button type="submit" class="btn-block-primary">Import</button>`;
    html += `</form>`;

    processorImporterFormDiv.innerHTML = html;
    processorImporterFormDiv.style.display = "block";

    const statusEl = processorImporterFormDiv.querySelector("#proc-imp-status");

    processorImporterFormDiv.querySelector("#proc-imp-form").addEventListener("submit", async (e) => {
      e.preventDefault();
      statusEl.textContent = "Importing\u2026";
      statusEl.style.color = "var(--text-muted)";

      const formEl = e.target;
      const hasFiles = importer.fields.some(f => f.field_type === "file");
      let body, headers = {};

      if (hasFiles) {
        body = new FormData(formEl);
      } else {
        const obj = { name: formEl.elements["name"].value };
        for (const field of importer.fields) {
          obj[field.key] = formEl.elements[field.key].value;
        }
        body = JSON.stringify(obj);
        headers["Content-Type"] = "application/json";
      }

      try {
        const res = await fetch(`/api/processor-importers/import/${encodeURIComponent(importer.name)}`, {
          method: "POST", headers, body,
        });
        const result = await res.json();
        if (res.ok) {
          let msg = `Imported "${result.name}" (${result.media_type})`;
          if (result.loaded) msg += `, ${result.loaded} files loaded`;
          statusEl.textContent = msg;
          statusEl.style.color = "var(--color-good)";
          setTimeout(() => {
            processorImporterModal.classList.remove("show");
            if (currentView === "dashboard") renderDashboardModels();
          }, 1500);
        } else {
          statusEl.textContent = result.error || "Import failed";
          statusEl.style.color = "var(--color-bad)";
        }
      } catch (err) {
        statusEl.textContent = `Error: ${err.message}`;
        statusEl.style.color = "var(--color-bad)";
      }
    });
  }

  function showProcessorLabelImporterForm(importer) {
    processorImporterList.style.display = "none";
    processorImporterBack.style.display = "inline-block";

    let html = `<h3 class="form-heading">${escapeHtml(importer.display_name)}</h3>`;
    html += `<p class="form-hint" style="margin-bottom:12px">Import labels and train a detector model.</p>`;
    html += `<form id="label-imp-form">`;
    // Name field (always required)
    html += `<div class="form-group">`;
    html += `<label class="form-label">Model Name *</label>`;
    html += `<input type="text" name="name" placeholder="e.g. Dog Barks" class="form-input" required>`;
    html += `<div class="form-hint">Name for the trained detector.</div>`;
    html += `</div>`;
    for (const field of importer.fields) {
      html += `<div class="form-group">`;
      html += `<label class="form-label">${escapeHtml(field.label)}${field.required ? " *" : ""}</label>`;
      if (field.field_type === "file") {
        html += `<input type="file" name="${escapeHtml(field.key)}" accept="${escapeHtml(field.accept)}" class="form-input" ${field.required ? "required" : ""}>`;
      } else if (field.field_type === "select") {
        html += `<select name="${escapeHtml(field.key)}" class="form-input">`;
        for (const opt of field.options) {
          html += `<option value="${escapeHtml(opt)}"${opt === field.default ? " selected" : ""}>${escapeHtml(opt || "(auto-detect)")}</option>`;
        }
        html += `</select>`;
      } else {
        const itype = field.field_type === "password" ? "password" : "text";
        const placeholder = escapeHtml(field.placeholder || field.description);
        html += `<input type="${itype}" name="${escapeHtml(field.key)}" value="${escapeHtml(field.default)}" placeholder="${placeholder}" class="form-input" ${field.required ? "required" : ""}>`;
      }
      if (field.description) {
        html += `<div class="form-hint">${escapeHtml(field.description)}</div>`;
      }
      html += `</div>`;
    }
    html += `<div id="label-imp-status" class="status-text compact"></div>`;
    html += `<button type="submit" class="btn-block-primary">Import & Train</button>`;
    html += `</form>`;

    processorImporterFormDiv.innerHTML = html;
    processorImporterFormDiv.style.display = "block";

    const statusEl = processorImporterFormDiv.querySelector("#label-imp-status");

    processorImporterFormDiv.querySelector("#label-imp-form").addEventListener("submit", async (e) => {
      e.preventDefault();
      statusEl.textContent = "Importing labels & training\u2026";
      statusEl.style.color = "var(--text-muted)";

      const formEl = e.target;
      const hasFiles = importer.fields.some(f => f.field_type === "file");
      let body, headers = {};

      if (hasFiles) {
        body = new FormData(formEl);
      } else {
        const obj = { name: formEl.elements["name"].value };
        for (const field of importer.fields) {
          obj[field.key] = formEl.elements[field.key].value;
        }
        body = JSON.stringify(obj);
        headers["Content-Type"] = "application/json";
      }

      try {
        const res = await fetch(`/api/autorun-detectors/from-label-import/${encodeURIComponent(importer.name)}`, {
          method: "POST", headers, body,
        });
        const result = await res.json();
        if (res.ok) {
          let msg = `Trained "${result.name}" (${result.media_type})`;
          if (result.loaded) msg += `, ${result.loaded} labels matched`;
          if (result.skipped) msg += `, ${result.skipped} skipped`;
          statusEl.textContent = msg;
          statusEl.style.color = "var(--color-good)";
          setTimeout(() => {
            processorImporterModal.classList.remove("show");
            if (currentView === "dashboard") renderDashboardModels();
          }, 1500);
        } else {
          statusEl.textContent = result.error || "Import failed";
          statusEl.style.color = "var(--color-bad)";
        }
      } catch (err) {
        statusEl.textContent = `Error: ${err.message}`;
        statusEl.style.color = "var(--color-bad)";
      }
    });
  }

  if (processorImporterModalClose) {
    processorImporterModalClose.addEventListener("click", () => {
      processorImporterModal.classList.remove("show");
      if (currentView === "dashboard") renderDashboardModels();
    });
  }

  if (processorImporterBack) {
    processorImporterBack.addEventListener("click", () => {
      processorImporterFormDiv.style.display = "none";
      processorImporterFormDiv.innerHTML = "";
      processorImporterBack.style.display = "none";
      processorImporterList.style.display = "";
    });
  }

  // ---- Progress tracking ----

  const smartIndicator = document.getElementById("smart-indicator");
  const stableIndicator = document.getElementById("stable-indicator");
  const spanIndicator = document.getElementById("span-indicator");
  const progressModal = document.getElementById("progress-modal");
  const modalClose = document.getElementById("modal-close");
  const goodCountSpan = document.getElementById("good-count");
  const badCountSpan = document.getElementById("bad-count");
  const labelingAnalysisModal = document.getElementById("labeling-analysis-modal");
  const labelingAnalysisBar = document.getElementById("labeling-analysis-bar");
  const labelingAnalysisText = document.getElementById("labeling-analysis-text");
  const labelingAnalysisPct = document.getElementById("labeling-analysis-pct");

  // Keep latest status data for span info display
  let _lastStatusData = null;

  // Update the learned-sort description with current vote counts
  function updateLearnedSortDesc() {
    learnedSortDesc.textContent = `${votes.good.length} G, ${votes.bad.length} B`;
  }

  // Update label counts and schedule an indicator refresh
  function updateLabelCounts() {
    goodCountSpan.textContent = `(${votes.good.length})`;
    badCountSpan.textContent = `(${votes.bad.length})`;
    updateLearnedSortDesc();
    scheduleLabelingStatusUpdate();
  }

  // ---- Labeling status indicator ----

  let _statusTimer = null;

  function scheduleLabelingStatusUpdate() {
    clearTimeout(_statusTimer);
    _statusTimer = setTimeout(fetchLabelingStatus, 1200);
  }

  async function fetchLabelingStatus() {
    try {
      const res = await fetch("/api/labeling-status");
      if (!res.ok) return;
      const data = await res.json();
      if (data.error) return;
      applyLabelingStatus(data);
    } catch (_) {
      // Silently ignore — the indicator will just stay in its last state
    }
  }

  function _applyIndicator(btn, subtextEl, metric) {
    btn.dataset.status = metric.status;
    if (metric.status === "red") {
      subtextEl.textContent = "";
    } else if (metric.status === "yellow") {
      subtextEl.textContent = "";
    } else if (metric.status === "green") {
      subtextEl.textContent = "";
    } else {
      subtextEl.textContent = "";
    }
  }

  function applyLabelingStatus(data) {
    _lastStatusData = data;
    if (data.smart) {
      _applyIndicator(smartIndicator, document.getElementById("smart-subtext"), data.smart);
    }
    if (data.stable) {
      _applyIndicator(stableIndicator, document.getElementById("stable-subtext"), data.stable);
    }
    if (data.span) {
      _applyIndicator(spanIndicator, document.getElementById("span-subtext"), data.span);
    }
    // Feed indicator statuses into the autopilot state machine
    _autopilotOnIndicatorUpdate(data);
  }

  // Generic handler: fetch full analysis, then show the relevant section
  async function showMetricDetail(metric, triggerBtn) {
    // Span with no good+bad votes: just show the text popup from cached status
    if (metric === "span" && (votes.good.length === 0 || votes.bad.length === 0)) {
      showSpanPopup();
      return;
    }

    if (votes.good.length === 0 || votes.bad.length === 0) {
      await vtAlert("Need at least one good and one bad vote to check progress", "warning");
      return;
    }

    pauseActiveMedia();

    // Show loading progress modal
    labelingAnalysisBar.style.width = "0%";
    labelingAnalysisPct.textContent = "0%";
    labelingAnalysisText.textContent = "Training models over label history…";
    labelingAnalysisModal.classList.add("show");

    let progress = 0;
    const progressInterval = setInterval(() => {
      progress += 3;
      if (progress > 90) progress = 90;
      labelingAnalysisBar.style.width = `${progress}%`;
      labelingAnalysisPct.textContent = `${progress}%`;
    }, 250);

    triggerBtn.disabled = true;
    const origLabel = triggerBtn.querySelector(".indicator-label").textContent;
    triggerBtn.querySelector(".indicator-label").textContent = "…";

    try {
      const res = await fetch("/api/labeling-progress", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      });

      clearInterval(progressInterval);
      labelingAnalysisBar.style.width = "100%";
      labelingAnalysisPct.textContent = "100%";
      labelingAnalysisText.textContent = "Done!";

      await new Promise(r => setTimeout(r, 350));
      labelingAnalysisModal.classList.remove("show");

      if (!res.ok) {
        const error = await res.json();
        await vtAlert(error.error || "Failed to analyze progress", "error");
        return;
      }

      const data = await res.json();
      displayProgressResults(data, metric);
      progressModal.classList.add("show");
    } catch (e) {
      clearInterval(progressInterval);
      labelingAnalysisModal.classList.remove("show");
      await vtAlert("Error analyzing progress: " + e.message, "error");
    } finally {
      triggerBtn.disabled = false;
      triggerBtn.querySelector(".indicator-label").textContent = origLabel;
      fetchLabelingStatus();
      if (!progressModal.classList.contains("show")) {
        resumeActiveMedia();
      }
    }
  }

  function showSpanPopup() {
    // Show the progress modal with only the Span section visible
    const sp = _lastStatusData && _lastStatusData.span ? _lastStatusData.span : null;
    const infoText = document.getElementById("span-info-text");
    if (!sp || sp.level < 0) {
      infoText.textContent = "No diversity tree coverage yet. Keep labeling diverse examples.";
    } else if (sp.level >= sp.depth) {
      infoText.textContent = `All ${sp.depth + 1} tree levels fully covered. Excellent diversity!`;
    } else {
      let msg = `Deepest full level: ${sp.level} of ${sp.depth}.`;
      if (sp.next_level_total > 0) {
        msg += ` Next level (${sp.level + 1}): ${sp.next_level_seen} of ${sp.next_level_total} nodes seen.`;
      }
      infoText.textContent = msg;
    }

    // Update stats from cached status data
    if (_lastStatusData) {
      document.getElementById("stat-total-labels").textContent = _lastStatusData.total_count || 0;
      document.getElementById("stat-total-medias").textContent = "—";
      const currentType = medias.length > 0 ? medias[0].type : null;
      const mtInfo = currentType ? mediaTypesMap[currentType] : null;
      document.getElementById("stat-total-medias-label").textContent = mtInfo ? `Total ${mtInfo.tab_title}` : "Total Medias";
    }

    document.getElementById("smart-section").style.display = "none";
    document.getElementById("stable-section").style.display = "none";
    document.getElementById("span-section").style.display = "";
    document.getElementById("progress-modal-title").textContent = "Diverse: Diversity Coverage";

    pauseActiveMedia();
    progressModal.classList.add("show");
  }

  smartIndicator.addEventListener("click", () => showMetricDetail("smart", smartIndicator));
  stableIndicator.addEventListener("click", () => showMetricDetail("stable", stableIndicator));
  spanIndicator.addEventListener("click", () => showMetricDetail("span", spanIndicator));

  modalClose.addEventListener("click", () => {
    progressModal.classList.remove("show");
    resumeActiveMedia();
  });

  progressModal.addEventListener("click", (e) => {
    if (e.target === progressModal) {
      progressModal.classList.remove("show");
      resumeActiveMedia();
    }
  });

  function displayProgressResults(data, metric) {
    // Update summary stats
    document.getElementById("stat-total-labels").textContent = data.total_labels;
    document.getElementById("stat-total-medias").textContent = data.total_medias;
    const currentType = medias.length > 0 ? medias[0].type : null;
    const mtInfo = currentType ? mediaTypesMap[currentType] : null;
    document.getElementById("stat-total-medias-label").textContent = mtInfo ? `Total ${mtInfo.tab_title}` : "Total Medias";

    const ecChart = document.getElementById("error-cost-chart");
    if (ecChart) ecChart.setAttribute("role", "img");
    const stChart = document.getElementById("stability-chart");
    if (stChart) stChart.setAttribute("role", "img");

    // Show only the relevant section
    const smartSec = document.getElementById("smart-section");
    const stableSec = document.getElementById("stable-section");
    const spanSec = document.getElementById("span-section");
    smartSec.style.display = "none";
    stableSec.style.display = "none";
    spanSec.style.display = "none";

    if (metric === "smart") {
      smartSec.style.display = "";
      renderErrorCostChart(data.error_cost_over_time);
      document.getElementById("progress-modal-title").textContent = "Smart: Error Cost Analysis";
      if (ecChart) {
        const lastCost = data.error_cost_over_time.length > 0
          ? data.error_cost_over_time[data.error_cost_over_time.length - 1].error_cost.toFixed(2)
          : "N/A";
        ecChart.setAttribute("aria-label", `Error cost chart with ${data.error_cost_over_time.length} data points. Latest error cost: ${lastCost}`);
      }
    } else if (metric === "stable") {
      stableSec.style.display = "";
      renderStabilityChart(data.stability_over_time);
      document.getElementById("progress-modal-title").textContent = "Stable: Prediction Flip Analysis";
      if (stChart) {
        const lastFlips = data.stability_over_time.length > 1
          ? data.stability_over_time[data.stability_over_time.length - 1].num_flips
          : "N/A";
        stChart.setAttribute("aria-label", `Prediction stability chart with ${data.stability_over_time.length} data points. Latest prediction flips: ${lastFlips}`);
      }
    } else if (metric === "span") {
      spanSec.style.display = "";
      renderDiversityChart(data.diversity_level_over_time);
      document.getElementById("progress-modal-title").textContent = "Diverse: Diversity Coverage";
      const dvChart = document.getElementById("diversity-chart");
      if (dvChart) {
        dvChart.setAttribute("role", "img");
        const lastLevel = data.diversity_level_over_time && data.diversity_level_over_time.length > 0
          ? data.diversity_level_over_time[data.diversity_level_over_time.length - 1].diversity_level.toFixed(2)
          : "N/A";
        dvChart.setAttribute("aria-label", `Diversity level chart with ${(data.diversity_level_over_time || []).length} data points. Latest level: ${lastLevel}`);
      }
      // Update span info text from cached status
      const sp = _lastStatusData && _lastStatusData.span ? _lastStatusData.span : null;
      const infoText = document.getElementById("span-info-text");
      if (!sp || sp.level < 0) {
        infoText.textContent = "No diversity tree coverage yet. Keep labeling diverse examples.";
      } else if (sp.level >= sp.depth) {
        infoText.textContent = `All ${sp.depth + 1} tree levels fully covered. Excellent diversity!`;
      } else {
        let msg = `Deepest full level: ${sp.level} of ${sp.depth}.`;
        if (sp.next_level_total > 0) {
          msg += ` Next level (${sp.level + 1}): ${sp.next_level_seen} of ${sp.next_level_total} nodes seen.`;
        }
        infoText.textContent = msg;
      }
    }

  }

  // Chart rendering delegated to static/charts.js (window.VTCharts)
  const renderErrorCostChart = window.VTCharts.renderErrorCostChart;
  const renderStabilityChart = window.VTCharts.renderStabilityChart;
  const renderDiversityChart = window.VTCharts.renderDiversityChart;


  // Modify fetchVotes to update label counts
  const originalFetchVotes = fetchVotes;
  fetchVotes = async function() {
    await originalFetchVotes();
    updateLabelCounts();
  };

  // Fetch media type metadata from the registry so the UI is data-driven.
  async function fetchMediaTypes() {
    try {
      const res = await fetch("/api/media-types");
      const data = await res.json();
      mediaTypesMap = {};
      (data.media_types || []).forEach(mt => { mediaTypesMap[mt.type_id] = mt; });
    } catch (_) {
      // Fallback: leave empty, the UI will degrade gracefully.
    }
  }

  // Initialize — go to dashboard; pre-fetch medias/votes if dataset is already loaded
  fetchMediaTypes().then(() => checkDatasetStatus()).then(async () => {
    if (datasetLoaded) {
      await fetchMedias();
      await fetchVotes();
    }
  });
  // ---- Dashboard event handlers ----

  // Header "Dashboard" button — visible during labeling
  if (headerDashboardBtn) {
    headerDashboardBtn.addEventListener("click", () => showDashboard());
  }

  // Burger menu "Dashboard" item
  if (menuDashboard) {
    menuDashboard.addEventListener("click", () => {
      if (menuDashboard.classList.contains("disabled")) return;
      closeBurgerMenu();
      showDashboard();
    });
  }

  // Dashboard: Add Dataset button — opens the dataset importer picker modal
  if (dashAddDatasetBtn) {
    dashAddDatasetBtn.addEventListener("click", () => openDatasetImporterPicker());
  }

  // Dataset importer picker modal — close / back buttons
  if (datasetImporterModalClose) {
    datasetImporterModalClose.addEventListener("click", () => {
      datasetImporterModal.classList.remove("show");
    });
  }
  if (datasetImporterBack) {
    datasetImporterBack.addEventListener("click", () => {
      datasetImporterFormDiv.style.display = "none";
      datasetImporterFormDiv.innerHTML = "";
      datasetImporterBack.style.display = "none";
      datasetImporterList.style.display = "";
    });
  }

  async function openDatasetImporterPicker() {
    // Reset to list view
    datasetImporterFormDiv.style.display = "none";
    datasetImporterFormDiv.innerHTML = "";
    datasetImporterBack.style.display = "none";
    datasetImporterList.style.display = "";

    // Fetch all importers
    let importers = [];
    try {
      const res = await fetch("/api/dataset/all-importers");
      if (res.ok) {
        const data = await res.json();
        importers = data.importers || [];
      }
    } catch (_) {}

    // Also add demo datasets as an option
    const options = importers.map(imp => `
      <div class="dataset-importer-option option-card" data-name="${escapeHtml(imp.name)}" data-type="importer">
        <span class="option-card-icon">${escapeHtml(imp.icon || '\uD83D\uDD0C')}</span>
        <div>
          <div class="option-card-title">${escapeHtml(imp.display_name)}</div>
          <div class="option-card-desc">${escapeHtml(imp.description)}</div>
        </div>
      </div>
    `).join("");

    // Add demo dataset option
    const demoOption = `
      <div class="dataset-importer-option option-card" data-name="demo" data-type="demo">
        <span class="option-card-icon">\uD83C\uDFC6</span>
        <div>
          <div class="option-card-title">Load Demo Dataset</div>
          <div class="option-card-desc">Choose from a selection of pre-configured demo datasets</div>
        </div>
      </div>
    `;

    datasetImporterList.innerHTML = options + demoOption;

    // Wire up click handlers
    datasetImporterList.querySelectorAll(".dataset-importer-option").forEach(el => {
      el.setAttribute("role", "button");
      el.setAttribute("tabindex", "0");
      const name = el.dataset.name;
      const type = el.dataset.type;
      el.addEventListener("click", () => {
        if (type === "demo") {
          showDashDemoDatasetPicker();
        } else if (name === "pickle") {
          // Pickle = file upload — close modal and trigger file input
          datasetImporterModal.classList.remove("show");
          dashFileInput.click();
        } else if (name === "combine_datasets") {
          // Combine = close modal and go to welcome screen combine flow
          datasetImporterModal.classList.remove("show");
          showCombineDatasetsForm();
          datasetWelcome.style.display = "flex";
          dashboardView.style.display = "none";
        } else {
          const imp = importers.find(i => i.name === name);
          if (imp) showDashImporterForm(imp);
        }
      });
      el.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") { e.preventDefault(); el.click(); }
      });
    });

    datasetImporterModal.classList.add("show");
  }

  // Show an importer's form inline within the dataset importer picker modal
  function showDashImporterForm(importer) {
    datasetImporterList.style.display = "none";
    datasetImporterBack.style.display = "";

    let html = `<h3 style="margin-bottom:12px;">${escapeHtml(importer.icon || '\uD83D\uDD0C')} ${escapeHtml(importer.display_name)}</h3>`;
    html += `<form id="dash-imp-form">`;
    for (const field of importer.fields) {
      html += `<div class="form-group">`;
      html += `<label class="form-label">${escapeHtml(field.label)}${field.required ? " *" : ""}</label>`;
      if (field.field_type === "file") {
        html += `<input type="file" name="${escapeHtml(field.key)}" accept="${escapeHtml(field.accept || "")}" class="form-input" ${field.required ? "required" : ""}>`;
      } else if (field.field_type === "select") {
        html += `<select name="${escapeHtml(field.key)}" class="form-input">`;
        for (const opt of field.options) {
          html += `<option value="${escapeHtml(opt)}"${opt === field.default ? " selected" : ""}>${escapeHtml(opt)}</option>`;
        }
        html += `</select>`;
      } else if (field.field_type === "folder") {
        html += `<div class="form-row"><input type="text" name="${escapeHtml(field.key)}" placeholder="${escapeHtml(field.description)}" class="form-input" style="flex:1;" data-folder-input="true" ${field.required ? "required" : ""}>`;
        html += `<button type="button" data-browse-btn="true" class="btn-browse">Browse\u2026</button></div>`;
        html += `<input type="file" data-folder-picker="true" webkitdirectory style="display:none;">`;
      } else {
        const itype = field.field_type === "url" ? "url" : "text";
        html += `<input type="${itype}" name="${escapeHtml(field.key)}" value="${escapeHtml(field.default || "")}" placeholder="${escapeHtml(field.description)}" class="form-input" ${field.required ? "required" : ""}>`;
      }
      if (field.description) {
        html += `<div class="form-hint">${escapeHtml(field.description)}</div>`;
      }
      html += `</div>`;
    }
    html += `<div id="dash-imp-status" class="status-text" style="min-height:1.2em; margin:8px 0;"></div>`;
    html += `<button type="submit" class="btn-block-primary">Import</button>`;
    html += `</form>`;

    datasetImporterFormDiv.innerHTML = html;
    datasetImporterFormDiv.style.display = "block";

    // Wire up folder browse buttons
    const browseBtn = datasetImporterFormDiv.querySelector("[data-browse-btn]");
    const folderPicker = datasetImporterFormDiv.querySelector("[data-folder-picker]");
    const folderTextInput = datasetImporterFormDiv.querySelector("[data-folder-input]");
    if (browseBtn && folderPicker && folderTextInput) {
      browseBtn.addEventListener("click", () => folderPicker.click());
      folderPicker.addEventListener("change", () => {
        if (folderPicker.files.length > 0) {
          const topFolder = folderPicker.files[0].webkitRelativePath.split("/")[0];
          if (!folderTextInput.value) {
            folderTextInput.placeholder = `Selected: ${topFolder} \u2014 enter full path below`;
          }
        }
      });
    }

    document.getElementById("dash-imp-form").addEventListener("submit", async (e) => {
      e.preventDefault();
      const formEl = e.target;
      const statusEl = document.getElementById("dash-imp-status");
      const hasFiles = importer.fields.some(f => f.field_type === "file");
      let body, headers = {};
      if (hasFiles) {
        body = new FormData(formEl);
      } else {
        const obj = {};
        for (const field of importer.fields) {
          obj[field.key] = formEl.elements[field.key].value;
        }
        body = JSON.stringify(obj);
        headers["Content-Type"] = "application/json";
      }

      statusEl.textContent = "Importing\u2026";
      statusEl.style.color = "var(--text-secondary)";

      // Close modal and show in-grid progress
      datasetImporterModal.classList.remove("show");
      showDashGridProgress("Importing\u2026");

      try {
        const res = await fetch(`/api/dataset/import/${importer.name}`, { method: "POST", headers, body });
        if (!res.ok) {
          const err = await res.json();
          if (dashProgressMessage) { dashProgressMessage.textContent = `Error: ${err.error}`; dashProgressMessage.style.color = "var(--color-bad)"; }
          return;
        }
      } catch (err) {
        if (dashProgressMessage) { dashProgressMessage.textContent = `Error: ${err.message}`; dashProgressMessage.style.color = "var(--color-bad)"; }
        return;
      }

      dashSelectedDataset = null;
      dashPendingAction = null;
      startDashProgressPolling();
    });
  }

  // Show demo dataset picker within the dataset importer modal
  async function showDashDemoDatasetPicker() {
    datasetImporterList.style.display = "none";
    datasetImporterBack.style.display = "";

    datasetImporterFormDiv.innerHTML = '<p style="color:var(--text-muted); padding:8px;">Loading demo datasets\u2026</p>';
    datasetImporterFormDiv.style.display = "block";

    let demos = [];
    try {
      const res = await fetch("/api/dataset/demo-list");
      if (res.ok) {
        const data = await res.json();
        demos = data.datasets || [];
      }
    } catch (_) {}

    if (demos.length === 0) {
      datasetImporterFormDiv.innerHTML = '<p style="color:var(--text-muted); padding:8px;">No demo datasets available.</p>';
      return;
    }

    // Group by media type
    const grouped = {};
    demos.forEach(ds => {
      const mt = ds.media_type || "audio";
      if (!grouped[mt]) grouped[mt] = [];
      grouped[mt].push(ds);
    });
    const mediaOrder = Object.keys(mediaTypesMap).filter(mt => (grouped[mt] || []).length > 0);

    const statusOrder = { ready: 0, needs_embedding: 1, needs_download: 2 };

    function buildStatusBadge(st) {
      if (st === "ready") return '<span class="ready-badge">Ready</span>';
      if (st === "needs_embedding") return '<span class="embedding-badge">Needs Embed</span>';
      return '<span class="download-badge">Needs Download</span>';
    }

    // Build tabs + table content
    const wrapper = document.createElement("div");

    const tabBar = document.createElement("div");
    tabBar.className = "demo-tab-bar";
    wrapper.appendChild(tabBar);

    const contentArea = document.createElement("div");
    contentArea.className = "demo-content-area";
    wrapper.appendChild(contentArea);

    const sections = {};

    const sortColumns = [
      { key: "label", label: "Name" },
      { key: "num_files", label: "# Media" },
      { key: "num_categories", label: "# Cat." },
      { key: "description", label: "Description" },
      { key: "status", label: "Readiness" },
    ];

    function renderTable(items, section) {
      const sortState = section._demoSort || { key: "label", asc: true };
      const sorted = [...items].sort((a, b) => {
        let va = a[sortState.key], vb = b[sortState.key];
        if (sortState.key === "status") { va = statusOrder[va] ?? 3; vb = statusOrder[vb] ?? 3; }
        if (typeof va === "number" && typeof vb === "number") return sortState.asc ? va - vb : vb - va;
        va = String(va || "").toLowerCase(); vb = String(vb || "").toLowerCase();
        return sortState.asc ? va.localeCompare(vb) : vb.localeCompare(va);
      });

      const tbody = section.querySelector("tbody");
      tbody.innerHTML = "";
      sorted.forEach(ds => {
        const st = ds.status || (ds.ready ? "ready" : "needs_download");
        const tr = document.createElement("tr");
        tr.className = "demo-row" + (st === "ready" ? " ready" : st === "needs_embedding" ? " needs-embedding" : "");
        tr.setAttribute("role", "button");
        tr.setAttribute("tabindex", "0");
        tr.setAttribute("aria-label", `${ds.label}: ${ds.description}`);
        tr.addEventListener("click", () => {
          datasetImporterModal.classList.remove("show");
          dashSelectedDataset = ds;
          // Load the selected demo dataset immediately from dashboard
          dashPendingAction = null;
          dashLoadSelectedDataset();
        });
        tr.addEventListener("keydown", (e) => {
          if (e.key === "Enter" || e.key === " ") { e.preventDefault(); tr.click(); }
        });
        const descShort = ds.description.length > 60 ? ds.description.slice(0, 57) + "\u2026" : ds.description;
        tr.innerHTML = `
          <td class="col-name">${escapeHtml(ds.label)}</td>
          <td class="col-num">${ds.num_files}</td>
          <td class="col-num">${ds.num_categories}</td>
          <td class="col-desc" title="${escapeHtml(ds.description)}">${escapeHtml(descShort)}</td>
          <td class="col-status">${buildStatusBadge(st)}</td>
        `;
        tbody.appendChild(tr);
      });

      section.querySelectorAll("th[data-sort]").forEach(th => {
        const arrow = th.querySelector(".sort-arrow");
        if (th.dataset.sort === sortState.key) {
          arrow.textContent = sortState.asc ? " \u25B2" : " \u25BC";
        } else {
          arrow.textContent = "";
        }
      });
    }

    // Pick initial tab
    let initialTab = mediaOrder[0] || Object.keys(mediaTypesMap)[0] || "audio";
    try {
      const settingsRes = await fetch("/api/settings");
      if (settingsRes.ok) {
        const settingsData = await settingsRes.json();
        const favs = settingsData.autoload_media_types || [];
        const firstFav = favs.find(f => mediaOrder.includes(f));
        if (firstFav) initialTab = firstFav;
      }
    } catch (_) {}

    mediaOrder.forEach(mt => {
      const items = grouped[mt];
      const mtInfo = mediaTypesMap[mt] || { icon: "\uD83D\uDCC1", tab_title: mt };

      const tab = document.createElement("button");
      tab.className = "demo-tab";
      tab.dataset.mediaType = mt;
      tab.textContent = `${mtInfo.icon} ${mtInfo.tab_title}`;
      tab.addEventListener("click", () => {
        tabBar.querySelectorAll(".demo-tab").forEach(t => t.classList.remove("active"));
        tab.classList.add("active");
        Object.keys(sections).forEach(k => {
          sections[k].style.display = k === mt ? "" : "none";
        });
      });
      tabBar.appendChild(tab);

      const section = document.createElement("div");
      section.className = "demo-section";
      section._demoSort = { key: "num_files", asc: true };
      section._mt = mt;
      section.style.display = "none";

      const headerRow = sortColumns.map(col =>
        `<th data-sort="${col.key}">${col.label}<span class="sort-arrow"></span></th>`
      ).join("");
      section.innerHTML = `<table class="demo-table"><thead><tr>${headerRow}</tr></thead><tbody></tbody></table>`;

      section.querySelectorAll("th[data-sort]").forEach(th => {
        th.addEventListener("click", () => {
          const key = th.dataset.sort;
          if (section._demoSort.key === key) {
            section._demoSort.asc = !section._demoSort.asc;
          } else {
            section._demoSort = { key, asc: true };
          }
          renderTable(items, section);
        });
      });

      renderTable(items, section);
      sections[mt] = section;
      contentArea.appendChild(section);
    });

    const initialTabBtn = tabBar.querySelector(`.demo-tab[data-media-type="${initialTab}"]`);
    if (initialTabBtn) {
      initialTabBtn.classList.add("active");
      if (sections[initialTab]) sections[initialTab].style.display = "";
    }

    datasetImporterFormDiv.innerHTML = "";
    datasetImporterFormDiv.appendChild(wrapper);
  }

  // Dashboard: File input for pickle upload (used by pickle importer option)
  if (dashFileInput) {
    dashFileInput.addEventListener("change", async () => {
      const file = dashFileInput.files[0];
      if (!file) return;

      showDashGridProgress("Uploading...");

      const formData = new FormData();
      formData.append("file", file);
      try {
        const res = await fetch("/api/dataset/load-file", { method: "POST", body: formData });
        if (!res.ok) {
          const data = await res.json();
          if (dashProgressMessage) { dashProgressMessage.textContent = `Error: ${data.error || "Upload failed"}`; dashProgressMessage.style.color = "var(--color-bad)"; }
          dashFileInput.value = "";
          return;
        }
      } catch (e) {
        if (dashProgressMessage) { dashProgressMessage.textContent = `Error: ${e.message}`; dashProgressMessage.style.color = "var(--color-bad)"; }
        dashFileInput.value = "";
        return;
      }

      dashFileInput.value = "";
      dashSelectedDataset = null;
      // Poll progress
      dashPendingAction = null;
      startDashProgressPolling();
    });
  }

  // Dashboard: Add Model button — opens a picker with New Model + processor importers
  if (dashAddModelBtn) {
    dashAddModelBtn.addEventListener("click", () => openAddModelPicker());
  }

  async function openAddModelPicker() {
    // Fetch processor importers and label importers for the importer options
    let importers = [];
    let labelImporters = [];
    try {
      const [procRes, labelRes] = await Promise.all([
        fetch("/api/processor-importers"),
        fetch("/api/label-importers"),
      ]);
      if (procRes.ok) importers = await procRes.json();
      if (labelRes.ok) labelImporters = await labelRes.json();
    } catch (_) { /* ignore */ }

    // Reset modal to list view
    processorImporterFormDiv.style.display = "none";
    processorImporterFormDiv.innerHTML = "";
    processorImporterBack.style.display = "none";
    processorImporterList.style.display = "";

    // Update modal title
    const modalTitle = document.getElementById("processor-importer-modal-title");
    if (modalTitle) modalTitle.textContent = "Add Model";

    // Build options: New Model first, then processor importers, then label importers
    let html = `
      <div class="processor-importer-option option-card" data-name="__new_model__" role="button" tabindex="0">
        <span class="option-card-icon">\u2795</span>
        <div>
          <div class="option-card-title">New Model</div>
          <div class="option-card-desc">Create a new model with a name and media type.</div>
        </div>
      </div>`;

    html += importers.map(imp => `
      <div class="processor-importer-option option-card" data-name="${escapeHtml(imp.name)}" role="button" tabindex="0">
        <span class="option-card-icon">${escapeHtml(imp.icon || '\u{1F9E9}')}</span>
        <div>
          <div class="option-card-title">${escapeHtml(imp.display_name)}</div>
          <div class="option-card-desc">${escapeHtml(imp.description)}</div>
        </div>
      </div>
    `).join("");

    html += labelImporters.map(imp => `
      <div class="processor-importer-option option-card" data-name="__label_imp__${escapeHtml(imp.name)}" role="button" tabindex="0">
        <span class="option-card-icon">${escapeHtml(imp.icon || '\uD83C\uDFF7\uFE0F')}</span>
        <div>
          <div class="option-card-title">${escapeHtml(imp.display_name)}</div>
          <div class="option-card-desc">Train model from labels: ${escapeHtml(imp.description)}</div>
        </div>
      </div>
    `).join("");

    processorImporterList.innerHTML = html;

    // Wire up click handlers
    processorImporterList.querySelectorAll(".processor-importer-option").forEach(el => {
      const name = el.dataset.name;
      el.addEventListener("click", () => {
        if (name === "__new_model__") {
          showNewModelForm();
        } else if (name.startsWith("__label_imp__")) {
          const impName = name.slice("__label_imp__".length);
          const imp = labelImporters.find(i => i.name === impName);
          if (imp) showProcessorLabelImporterForm(imp);
        } else {
          const imp = importers.find(i => i.name === name);
          if (imp) showProcessorImporterForm(imp);
        }
      });
      el.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") { e.preventDefault(); el.click(); }
      });
    });

    processorImporterModal.classList.add("show");
  }

  async function showNewModelForm() {
    processorImporterList.style.display = "none";
    processorImporterBack.style.display = "inline-block";

    // Fetch importers for the example type dropdown
    let procImporters = [];
    let lblImporters = [];
    try {
      const [procRes, lblRes] = await Promise.all([
        fetch("/api/processor-importers"),
        fetch("/api/label-importers"),
      ]);
      if (procRes.ok) procImporters = await procRes.json();
      if (lblRes.ok) lblImporters = await lblRes.json();
    } catch (_) { /* ignore */ }

    // Guess the media type: prefer single dataset type, then single autoload type
    let guessedMediaType = "";
    const datasetTypes = [...new Set(dashRegisteredDatasets.map(d => d.media_type).filter(Boolean))];
    if (datasetTypes.length === 1) {
      guessedMediaType = datasetTypes[0];
    } else {
      try {
        const sRes = await fetch("/api/settings");
        if (sRes.ok) {
          const sData = await sRes.json();
          const autoloads = sData.autoload_media_types || [];
          if (autoloads.length === 1) guessedMediaType = autoloads[0];
        }
      } catch (_) { /* ignore */ }
    }

    // Build media type options from the registry
    const mtOptions = Object.entries(mediaTypesMap).map(([id, mt]) =>
      `<option value="${escapeHtml(id)}"${id === guessedMediaType ? " selected" : ""}>${escapeHtml(mt.icon || "")} ${escapeHtml(mt.name || id)}</option>`
    ).join("");

    // Build example type options: built-in + importers
    let exTypeOptions = `<option value="text">Text description</option>`;
    exTypeOptions += `<option value="media">Server-side example</option>`;
    exTypeOptions += `<option value="detector">Detector</option>`;
    for (const imp of procImporters) {
      exTypeOptions += `<option value="proc_imp:${escapeHtml(imp.name)}">${escapeHtml(imp.icon || '\u{1F9E9}')} ${escapeHtml(imp.display_name)}</option>`;
    }
    for (const imp of lblImporters) {
      exTypeOptions += `<option value="label_imp:${escapeHtml(imp.name)}">${escapeHtml(imp.icon || '\uD83C\uDFF7\uFE0F')} ${escapeHtml(imp.display_name)}</option>`;
    }

    let html = ``;
    html += `<form id="new-model-form">`;
    html += `<div class="form-group">`;
    html += `<label class="form-label">Model Name *</label>`;
    html += `<input type="text" name="name" placeholder="e.g. Dog Barks" class="form-input" required>`;
    html += `<div class="form-hint">A short name for this model.</div>`;
    html += `</div>`;
    html += `<div class="form-group">`;
    html += `<label class="form-label">Media Type *</label>`;
    html += `<select name="media_type" class="form-input" required>`;
    html += mtOptions || `<option value="audio">Audio</option><option value="image">Image</option><option value="paragraph">Text</option><option value="video">Video</option>`;
    html += `</select>`;
    html += `<div class="form-hint">The type of media this model will be trained on.</div>`;
    html += `</div>`;
    html += `<div class="form-group">`;
    html += `<label class="form-label">Examples *</label>`;
    html += `<div id="new-model-examples-grid" class="examples-grid" style="min-height:36px;margin-bottom:6px"></div>`;
    html += `<div class="examples-add-bar">`;
    html += `<select id="new-model-example-type" class="form-select-inline">`;
    html += exTypeOptions;
    html += `</select>`;
    html += `<button type="button" id="new-model-example-add" class="btn-sm">+ Add</button>`;
    html += `</div>`;
    html += `<div class="form-hint">Add at least one example so the model knows what to find.</div>`;
    html += `</div>`;
    html += `<div id="new-model-status" class="status-text compact"></div>`;
    html += `<button type="submit" id="new-model-ok-btn" class="btn-block-primary" disabled>Ok</button>`;
    html += `</form>`;

    processorImporterFormDiv.innerHTML = html;
    processorImporterFormDiv.style.display = "block";

    const statusEl = processorImporterFormDiv.querySelector("#new-model-status");
    const okBtn = processorImporterFormDiv.querySelector("#new-model-ok-btn");
    const examplesGrid = processorImporterFormDiv.querySelector("#new-model-examples-grid");
    const exampleTypeSelect = processorImporterFormDiv.querySelector("#new-model-example-type");
    const exampleAddBtn = processorImporterFormDiv.querySelector("#new-model-example-add");
    const nameInput = processorImporterFormDiv.querySelector("input[name='name']");

    // Track examples locally
    let newModelExamples = [];

    function refreshNewModelGrid() {
      renderExamplesGrid(examplesGrid, newModelExamples, (updated) => {
        newModelExamples = updated;
        refreshNewModelGrid();
      });
      updateOkBtn();
    }

    function updateOkBtn() {
      const name = nameInput ? nameInput.value.trim() : "";
      okBtn.disabled = !(name && newModelExamples.length > 0);
    }

    if (nameInput) nameInput.addEventListener("input", updateOkBtn);

    // Initial render
    refreshNewModelGrid();

    exampleAddBtn.addEventListener("click", async () => {
      const type = exampleTypeSelect.value;
      const ex = await promptForExample(type);
      if (ex) {
        newModelExamples = [...newModelExamples, ex];
        refreshNewModelGrid();
      }
    });

    processorImporterFormDiv.querySelector("#new-model-form").addEventListener("submit", async (e) => {
      e.preventDefault();
      const formEl = e.target;
      const name = formEl.elements["name"].value.trim();
      const media_type = formEl.elements["media_type"].value;

      if (!name) {
        statusEl.textContent = "Name is required";
        statusEl.style.color = "var(--color-bad)";
        return;
      }
      if (newModelExamples.length === 0) {
        statusEl.textContent = "Add at least one example";
        statusEl.style.color = "var(--color-bad)";
        return;
      }

      statusEl.textContent = "Creating\u2026";
      statusEl.style.color = "var(--text-muted)";
      okBtn.disabled = true;

      try {
        const res = await fetch("/api/autorun-detectors", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ name, media_type, examples: newModelExamples }),
        });
        const result = await res.json();
        if (res.ok) {
          statusEl.textContent = `Created "${name}"`;
          statusEl.style.color = "var(--color-good)";
          setTimeout(() => {
            processorImporterModal.classList.remove("show");
            if (currentView === "dashboard") renderDashboardModels();
          }, 800);
        } else {
          statusEl.textContent = result.error || "Failed to create model";
          statusEl.style.color = "var(--color-bad)";
          okBtn.disabled = false;
        }
      } catch (err) {
        statusEl.textContent = `Error: ${err.message}`;
        statusEl.style.color = "var(--color-bad)";
        okBtn.disabled = false;
      }
    });
  }

  // ---- Examples Editor Modal ----

  let _examplesEditorTarget = null; // The detector object being edited

  function openExamplesEditorModal(det) {
    _examplesEditorTarget = det;
    const localExamples = [...(det.examples || [])];
    if (examplesEditorStatus) examplesEditorStatus.textContent = "";

    function refresh() {
      renderExamplesGrid(examplesEditorGrid, localExamples, (updated) => {
        localExamples.length = 0;
        localExamples.push(...updated);
        refresh();
      });
    }
    refresh();
    examplesEditorModal.classList.add("show");

    // Populate editor type dropdown with importer options (once)
    if (examplesEditorType && !examplesEditorType._importersLoaded) {
      examplesEditorType._importersLoaded = true;
      Promise.all([
        fetch("/api/processor-importers").then(r => r.ok ? r.json() : []).catch(() => []),
        fetch("/api/label-importers").then(r => r.ok ? r.json() : []).catch(() => []),
      ]).then(([procImps, lblImps]) => {
        for (const imp of procImps) {
          const opt = document.createElement("option");
          opt.value = `proc_imp:${imp.name}`;
          opt.textContent = `${imp.icon || '\u{1F9E9}'} ${imp.display_name}`;
          examplesEditorType.appendChild(opt);
        }
        for (const imp of lblImps) {
          const opt = document.createElement("option");
          opt.value = `label_imp:${imp.name}`;
          opt.textContent = `${imp.icon || '\uD83C\uDFF7\uFE0F'} ${imp.display_name}`;
          examplesEditorType.appendChild(opt);
        }
      });
    }

    // Wire Add button (replace handler to avoid stacking)
    const addHandler = async () => {
      const type = examplesEditorType.value;
      const ex = await promptForExample(type);
      if (ex) {
        localExamples.push(ex);
        refresh();
      }
    };
    examplesEditorAdd._handler && examplesEditorAdd.removeEventListener("click", examplesEditorAdd._handler);
    examplesEditorAdd._handler = addHandler;
    examplesEditorAdd.addEventListener("click", addHandler);

    // Wire Save button
    const saveHandler = async () => {
      if (examplesEditorStatus) {
        examplesEditorStatus.textContent = "Saving\u2026";
        examplesEditorStatus.style.color = "var(--text-muted)";
      }
      try {
        const url = det.trainable
          ? `/api/trainable-models/${encodeURIComponent(det.name)}/examples`
          : `/api/autorun-detectors/${encodeURIComponent(det.name)}/examples`;
        const res = await fetch(url, {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ examples: localExamples }),
        });
        if (res.ok) {
          det.examples = [...localExamples];
          if (examplesEditorStatus) {
            examplesEditorStatus.textContent = "Saved";
            examplesEditorStatus.style.color = "var(--color-good)";
          }
          // Update the autopilot panel if this model is the active train model
          if (_dashboardTrainMode && _dashboardTrainMode.model && _dashboardTrainMode.model.name === det.name) {
            _dashboardTrainMode.model.examples = [...localExamples];
            refreshAutopilotExamples();
          }
          setTimeout(() => {
            examplesEditorModal.classList.remove("show");
            if (currentView === "dashboard") renderDashboardModels();
          }, 600);
        } else {
          const data = await res.json();
          if (examplesEditorStatus) {
            examplesEditorStatus.textContent = data.error || "Save failed";
            examplesEditorStatus.style.color = "var(--color-bad)";
          }
        }
      } catch (err) {
        if (examplesEditorStatus) {
          examplesEditorStatus.textContent = `Error: ${err.message}`;
          examplesEditorStatus.style.color = "var(--color-bad)";
        }
      }
    };
    examplesEditorSave._handler && examplesEditorSave.removeEventListener("click", examplesEditorSave._handler);
    examplesEditorSave._handler = saveHandler;
    examplesEditorSave.addEventListener("click", saveHandler);
  }

  if (examplesEditorModalClose) {
    examplesEditorModalClose.addEventListener("click", () => {
      examplesEditorModal.classList.remove("show");
    });
  }

  // Dashboard: Label button
  if (dashLabelBtn) {
    dashLabelBtn.addEventListener("click", async () => {
      // Get the selected model from registry
      const selMs = dashRegisteredModels.filter(m => dashSelectedModelIds.includes(m.id));
      if (selMs.length === 1 && selMs[0].trainable) {
        // Build a model object compatible with _dashboardTrainMode
        const regModel = selMs[0];
        _dashboardTrainMode = {
          model: {
            name: regModel.trainable_model_name || regModel.name,
            text_query: regModel.text_query || "",
            trainable: true,
          },
        };

        // Fetch examples from the trainable model or autorun detector so
        // that Autopilot mode has the data it needs to start.
        const modelName = _dashboardTrainMode.model.name;
        try {
          const tmRes = await fetch(`/api/trainable-models/${encodeURIComponent(modelName)}`);
          if (tmRes.ok) {
            const tmData = await tmRes.json();
            _dashboardTrainMode.model.examples = tmData.examples || [];
            if (!_dashboardTrainMode.model.text_query && tmData.text_query) {
              _dashboardTrainMode.model.text_query = tmData.text_query;
            }
          }
        } catch (_) { /* ignore */ }
        // Fallback: try the autorun detector if no examples yet
        if (!_dashboardTrainMode.model.examples || _dashboardTrainMode.model.examples.length === 0) {
          const detName = regModel.detector_name || regModel.name;
          try {
            const detRes = await fetch(`/api/autorun-detectors/${encodeURIComponent(detName)}/examples`);
            if (detRes.ok) {
              const detData = await detRes.json();
              _dashboardTrainMode.model.examples = detData.examples || [];
            }
          } catch (_) { /* ignore */ }
        }
      } else {
        _dashboardTrainMode = null;
      }

      // Get the selected dataset from registry
      const selDs = dashRegisteredDatasets.filter(d => dashSelectedDatasetIds.includes(d.id));

      if (selDs.length === 1) {
        if (selDs[0].loaded) {
          // Already loaded — set up labeling (import labels, text sort, etc.)
          await fetchMedias();

          if (_dashboardTrainMode && _dashboardTrainMode.model) {
            // Clear any votes from a previous session so they don't
            // contaminate this model's labelset.
            try { await fetch("/api/votes/clear", { method: "POST" }); } catch (_) {}

            // Import labels from the trainable model's labelset
            try {
              const modelRes = await fetch(`/api/trainable-models/${encodeURIComponent(_dashboardTrainMode.model.name)}`);
              if (modelRes.ok) {
                const modelData = await modelRes.json();
                const labels = modelData.labelset && modelData.labelset.labels;
                if (labels && labels.length > 0) {
                  await fetch("/api/labels/import", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ labels }),
                  });
                }
              }
            } catch (_) { /* ignore label import errors */ }

            await fetchVotes();

            // Set text sort mode and trigger sort with the model's first example or text_query
            const exQuery = _dashboardTrainMode.model.text_query
              || (_dashboardTrainMode.model.examples && _dashboardTrainMode.model.examples.length > 0 && _dashboardTrainMode.model.examples[0].type === "text" ? _dashboardTrainMode.model.examples[0].value : "");
            sortMode = "text";
            textSortWrap.style.display = "";
            learnedSortWrap.style.display = "none";
            loadSortWrap.style.display = "none";
            document.querySelectorAll('input[name="sort-mode"]').forEach(r => {
              r.checked = r.value === "text";
            });
            textSortInput.value = exQuery;
            if (exQuery) {
              fetchTextSort(exQuery);
            }
          }

          showMainUI();
          if (_dashboardTrainMode && tabAutopilot) {
            tabAutopilot.click();
          }
          if (medias.length > 0 && !selected) {
            selectMedia(medias[0].id);
          }
        } else {
          // Load from registry, then go to labeling
          showDashGridProgress("Loading dataset...");
          try {
            await fetch(`/api/datasets/registry/${encodeURIComponent(selDs[0].id)}/load`, { method: "POST" });
          } catch (_) {}
          const savedTrainMode = _dashboardTrainMode;
          dashPendingAction = async () => {
            _dashboardTrainMode = savedTrainMode;

            if (_dashboardTrainMode && _dashboardTrainMode.model) {
              // Import labels from the trainable model's labelset
              try {
                const modelRes = await fetch(`/api/trainable-models/${encodeURIComponent(_dashboardTrainMode.model.name)}`);
                if (modelRes.ok) {
                  const modelData = await modelRes.json();
                  const labels = modelData.labelset && modelData.labelset.labels;
                  if (labels && labels.length > 0) {
                    await fetch("/api/labels/import", {
                      method: "POST",
                      headers: { "Content-Type": "application/json" },
                      body: JSON.stringify({ labels }),
                    });
                  }
                }
              } catch (_) { /* ignore label import errors */ }

              await fetchVotes();

              // Set text sort mode and trigger sort with the model's first example or text_query
              const exQuery = _dashboardTrainMode.model.text_query
                || (_dashboardTrainMode.model.examples && _dashboardTrainMode.model.examples.length > 0 && _dashboardTrainMode.model.examples[0].type === "text" ? _dashboardTrainMode.model.examples[0].value : "");
              sortMode = "text";
              textSortWrap.style.display = "";
              learnedSortWrap.style.display = "none";
              loadSortWrap.style.display = "none";
              document.querySelectorAll('input[name="sort-mode"]').forEach(r => {
                r.checked = r.value === "text";
              });
              textSortInput.value = exQuery;
              if (exQuery) {
                fetchTextSort(exQuery);
              }
            }

            showMainUI();
            if (medias.length > 0 && !selected) {
              selectMedia(medias[0].id);
            }
          };
          startDashProgressPolling();
        }
        return;
      }

      // Fallback to old behavior for demo dataset selection
      if (dashSelectedDataset) {
        dashLoadSelectedDataset(() => {
          showMainUI();
          if (_dashboardTrainMode && tabAutopilot) {
            tabAutopilot.click();
          }
          if (medias.length > 0 && !selected) {
            selectMedia(medias[0].id);
          }
        });
      }
    });
  }

  // Dashboard: Find button
  if (dashDetectBtn) {
    dashDetectBtn.addEventListener("click", async () => {
      const selDs = dashRegisteredDatasets.filter(d => dashSelectedDatasetIds.includes(d.id));
      const selMs = dashRegisteredModels.filter(m => dashSelectedModelIds.includes(m.id));

      if (selDs.length === 0 && selMs.length === 0) {
        // Fallback to old single-detector behavior
        if (!dashSelectedDetector) {
          await vtAlert("Select a model from the Model grid first.", "warning");
          return;
        }
        async function runDetectLegacy() {
          autodetectProgressModal.classList.add("show");
          autodetectProgressText.textContent = "Running Find...";
          autodetectProgressBar.style.width = "0%";
          let progress = 0;
          const progressInterval = setInterval(() => {
            progress += 5;
            if (progress > 90) progress = 90;
            autodetectProgressBar.style.width = `${progress}%`;
          }, 200);
          const res = await fetch("/api/auto-detect", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ detector_name: dashSelectedDetector }),
          });
          clearInterval(progressInterval);
          autodetectProgressBar.style.width = "100%";
          setTimeout(async () => {
            autodetectProgressModal.classList.remove("show");
            if (!res.ok) {
              await vtAlert("Find failed.", "error");
              return;
            }
            res.json().then(data => displayAutodetectResults(data));
          }, 500);
        }
        if (datasetLoaded) {
          await runDetectLegacy();
        } else if (dashSelectedDataset) {
          dashLoadSelectedDataset(() => runDetectLegacy());
        }
        return;
      }

      // Multi-dataset, multi-model Find via /api/find
      autodetectProgressModal.classList.add("show");
      autodetectProgressText.textContent = "Running Find across datasets and models...";
      autodetectProgressBar.style.width = "0%";
      let progress = 0;
      const progressInterval = setInterval(() => {
        progress += 3;
        if (progress > 90) progress = 90;
        autodetectProgressBar.style.width = `${progress}%`;
      }, 300);

      try {
        const res = await fetch("/api/find", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            dataset_ids: selDs.map(d => d.id),
            model_ids: selMs.map(m => m.id),
          }),
        });

        clearInterval(progressInterval);
        autodetectProgressBar.style.width = "100%";

        setTimeout(async () => {
          autodetectProgressModal.classList.remove("show");
          if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            await vtAlert(err.error || "Find failed.", "error");
            return;
          }
          const data = await res.json();
          displayFindResults(data);
        }, 500);
      } catch (e) {
        clearInterval(progressInterval);
        autodetectProgressModal.classList.remove("show");
        await vtAlert("Find failed: " + e.message, "error");
      }
    });
  }

  // ---- Settings persistence ----

  function saveVolume(vol) {
    if (volumeSaveTimer) clearTimeout(volumeSaveTimer);
    volumeSaveTimer = setTimeout(() => {
      fetch("/api/settings", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ volume: vol }),
      }).catch(() => {});
    }, 500);
  }

  // ---- Theme toggle ----

  const themeBtns = document.querySelectorAll(".theme-btn");

  function applyTheme(theme) {
    if (theme === "light" || theme === "highviz") {
      document.documentElement.setAttribute("data-theme", theme);
    } else {
      document.documentElement.removeAttribute("data-theme");
    }
    themeBtns.forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.theme === theme);
    });
  }

  function saveTheme(theme) {
    fetch("/api/settings", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ theme: theme }),
    }).catch(() => {});
  }

  themeBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      const theme = btn.dataset.theme;
      applyTheme(theme);
      saveTheme(theme);
      const themeNames = { light: "Light", dark: "Dark", highviz: "High Visibility" };
      announce(`${themeNames[theme] || theme} mode enabled`);
    });
  });

  async function loadSettings() {
    try {
      const res = await fetch("/api/settings");
      if (!res.ok) return;
      const data = await res.json();
      if (typeof data.volume === "number") {
        audioVolume = data.volume;
        const audioEl = document.getElementById("media-audio");
        if (audioEl) audioEl.volume = audioVolume;
      }
      if (data.theme) {
        applyTheme(data.theme);
      }
      if (enrichDescCheckbox) {
        enrichDescCheckbox.checked = !!data.enrich_descriptions;
      }
      if (typeof data.inclusion === "number") {
        inclusion = data.inclusion;
        inclusionSlider.value = inclusion;
        inclusionValue.textContent = inclusion;
      }
      if (calibrateCountInput && typeof data.calibrate_count === "number") {
        calibrateCountInput.value = data.calibrate_count;
      }
      if (calibrationFractionInput && typeof data.calibration_fraction === "number") {
        calibrationFractionInput.value = data.calibration_fraction;
      }
      if (safeThresholdsCheckbox) {
        safeThresholdsCheckbox.checked = !!data.safe_thresholds;
      }
      if (data.swipe_animation !== undefined) {
        swipeAnimation = !!data.swipe_animation;
        if (swipeAnimationCheckbox) swipeAnimationCheckbox.checked = swipeAnimation;
      }
      if (showThumbnailsLeftCheckbox) {
        showThumbnailsLeftCheckbox.checked = !!data.show_thumbnails_left;
        showThumbnailsLeft = !!data.show_thumbnails_left;
      }
      if (showThumbnailsRightCheckbox) {
        const val = data.show_thumbnails_right !== undefined ? !!data.show_thumbnails_right : true;
        showThumbnailsRightCheckbox.checked = val;
        showThumbnailsRight = val;
      }
      if (autopilotTopGreensInput && typeof data.autopilot_top_greens === "number") {
        autopilotTopGreensInput.value = data.autopilot_top_greens;
      }
      if (autopilotHardRedsInput && typeof data.autopilot_hard_reds === "number") {
        autopilotHardRedsInput.value = data.autopilot_hard_reds;
      }
    } catch (_) {
      // Settings not available yet; use defaults
    }
  }

  updateLabelCounts();
  fetchLabelingStatus();
  loadSettings();

  // ---- Modal Escape key handler ----
  document.addEventListener("keydown", (e) => {
    if (e.key !== "Escape") return;

    // VT dialog gets priority (it's an alertdialog)
    if (vtDialogModal.classList.contains("show")) return;

    // Close any open modal on Escape, from most specific to least
    const modalClosePairs = [
      [examplesEditorModal, examplesEditorModalClose],
      [labelImporterModal, labelImporterModalClose],
      [labelExporterModal, labelExporterModalClose],
      [detectorExportModal, detectorExportModalClose],
      [processorImporterModal, processorImporterModalClose],
      [window.VTResults.getAutodetectModal(), window.VTResults.getAutodetectModalClose()],
      [datasetImporterModal, datasetImporterModalClose],
      [loadSortModal, loadSortModalClose],
      [settingsModal, settingsModalClose],
      [progressModal, modalClose],
    ];
    for (const [modal, closeBtn] of modalClosePairs) {
      if (modal && modal.classList.contains("show")) {
        e.preventDefault();
        if (closeBtn) closeBtn.click();
        else modal.classList.remove("show");
        return;
      }
    }
  });

  // ---- Keyboard shortcuts ----
  // Arrow Left / Right  = vote Bad / Good (like Tinder swipe)
  // Arrow Up / Down     = volume up / down
  // Spacebar            = pause / resume media (only when not typing)

  function isTyping() {
    const el = document.activeElement;
    if (!el) return false;
    const tag = el.tagName;
    if (tag === "INPUT" && el.type !== "checkbox" && el.type !== "radio" && el.type !== "range") return true;
    if (tag === "TEXTAREA" || tag === "SELECT") return true;
    if (el.isContentEditable) return true;
    return false;
  }

  document.addEventListener("keydown", (e) => {
    // Never intercept keys when any modal is open (dialogs, autodetect
    // results, settings, load-sort, etc.).  All modals use .modal.show.
    if (document.querySelector(".modal.show")) return;

    // Never intercept keys when typing in text fields
    if (isTyping()) return;

    // Never intercept if modifier keys are held (let browser/OS shortcuts work)
    if (e.ctrlKey || e.metaKey || e.altKey) return;

    switch (e.key) {
      case "ArrowRight":
        e.preventDefault();
        if (selected != null) castVote(selected, "good");
        break;
      case "ArrowLeft":
        e.preventDefault();
        if (selected != null) castVote(selected, "bad");
        break;
      case "ArrowUp":
        e.preventDefault();
        adjustVolume(0.05);
        break;
      case "ArrowDown":
        e.preventDefault();
        adjustVolume(-0.05);
        break;
      case " ":
        e.preventDefault();
        toggleMediaPlayback();
        break;
    }
  });

  function adjustVolume(delta) {
    audioVolume = Math.max(0, Math.min(1, audioVolume + delta));
    const audioEl = document.getElementById("media-audio");
    const videoEl = document.getElementById("media-video");
    if (audioEl) audioEl.volume = audioVolume;
    if (videoEl) videoEl.volume = audioVolume;
    saveVolume(audioVolume);
  }

  function toggleMediaPlayback() {
    const audioEl = document.getElementById("media-audio");
    const videoEl = document.getElementById("media-video");
    const media = audioEl || videoEl;
    if (!media) return;
    if (media.paused) {
      media.play().catch(() => {});
    } else {
      media.pause();
    }
  }

})();
