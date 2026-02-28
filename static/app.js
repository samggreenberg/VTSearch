(function() {
  let medias = [];
  let votes = { good: [], bad: [], click_times: {}, learned_scores: {} };
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

  // --- Custom VTSearch dialog system (replaces native alert/confirm/prompt) ---
  const vtDialogModal = document.getElementById("vt-dialog-modal");
  const vtDialogIcon = document.getElementById("vt-dialog-icon");
  const vtDialogMessage = document.getElementById("vt-dialog-message");
  const vtDialogInput = document.getElementById("vt-dialog-input");
  const vtDialogActions = document.getElementById("vt-dialog-actions");

  const VT_ICONS = {
    warning: "\u26A0\uFE0F",
    error: "\u274C",
    success: "\u2705",
    info: "\u2139\uFE0F",
  };

  function vtShowDialog({ message, type, showInput, inputDefault, buttons }) {
    return new Promise((resolve) => {
      vtDialogIcon.textContent = VT_ICONS[type] || VT_ICONS.info;
      vtDialogIcon.className = "vt-dialog-icon " + (type || "info");
      vtDialogMessage.textContent = message;

      if (showInput) {
        vtDialogInput.style.display = "";
        vtDialogInput.value = inputDefault || "";
      } else {
        vtDialogInput.style.display = "none";
      }

      function closeWith(value) {
        document.removeEventListener("keydown", keyHandler);
        vtDialogModal.classList.remove("show");
        resolve(value);
      }

      function keyHandler(e) {
        if (!vtDialogModal.classList.contains("show")) return;
        if (e.key === "Enter") {
          e.preventDefault();
          const primaryBtn = buttons.find((b) => b.primary);
          if (primaryBtn) closeWith(primaryBtn.value === "input" ? vtDialogInput.value : primaryBtn.value);
        } else if (e.key === "Escape") {
          e.preventDefault();
          const cancelBtn = buttons.find((b) => !b.primary);
          if (cancelBtn) closeWith(cancelBtn.value === "input" ? vtDialogInput.value : cancelBtn.value);
          else closeWith(buttons[0].value === "input" ? vtDialogInput.value : buttons[0].value);
        }
      }

      vtDialogActions.innerHTML = "";
      buttons.forEach((btn) => {
        const el = document.createElement("button");
        el.className = "vt-dialog-btn " + (btn.primary ? "primary" : "secondary");
        el.textContent = btn.label;
        el.addEventListener("click", () => {
          closeWith(btn.value === "input" ? vtDialogInput.value : btn.value);
        });
        vtDialogActions.appendChild(el);
      });

      document.addEventListener("keydown", keyHandler);
      vtDialogModal.classList.add("show");
      if (showInput) {
        setTimeout(() => vtDialogInput.focus(), 50);
      }
    });
  }

  function vtAlert(message, type) {
    type = type || "info";
    return vtShowDialog({
      message,
      type,
      showInput: false,
      buttons: [{ label: "OK", primary: true, value: true }],
    });
  }

  function vtConfirm(message, type) {
    type = type || "warning";
    return vtShowDialog({
      message,
      type,
      showInput: false,
      buttons: [
        { label: "Cancel", primary: false, value: false },
        { label: "OK", primary: true, value: true },
      ],
    });
  }

  function vtPrompt(message, defaultValue, type) {
    type = type || "info";
    return vtShowDialog({
      message,
      type,
      showInput: true,
      inputDefault: defaultValue || "",
      buttons: [
        { label: "Cancel", primary: false, value: null },
        { label: "OK", primary: true, value: "input" },
      ],
    });
  }

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

  // Burger menu elements
  const burgerBtn = document.getElementById("burger-btn");
  const burgerDropdown = document.getElementById("burger-dropdown");
  const menuDatasetChange = document.getElementById("menu-dataset-change");
  const menuLabelsExport = document.getElementById("menu-labels-export");
  const menuLabelsImport = document.getElementById("menu-labels-import");
  const menuLabelsStatus = document.getElementById("menu-labels-status");
  const menuDetectorImport = document.getElementById("menu-detector-import");
  const menuDetectorExport = document.getElementById("menu-detector-export");
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
  const autodetectModal = document.getElementById("autodetect-modal");
  const autodetectModalClose = document.getElementById("autodetect-modal-close");
  const autodetectSummary = document.getElementById("autodetect-summary");
  const autodetectResults = document.getElementById("autodetect-results");
  const copyResultsBtn = document.getElementById("copy-results-btn");
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
  const autopilotExamplesGrid = document.getElementById("autopilot-examples-grid");
  const autopilotExampleType = document.getElementById("autopilot-example-type");
  const autopilotExampleAdd = document.getElementById("autopilot-example-add");
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
  const dashChangeDatasetBtn = document.getElementById("dash-change-dataset-btn");
  const dashAddModelBtn = document.getElementById("dash-add-model-btn");
  const datasetImporterModal = document.getElementById("dataset-importer-modal");
  const datasetImporterModalClose = document.getElementById("dataset-importer-modal-close");
  const datasetImporterList = document.getElementById("dataset-importer-list");
  const datasetImporterFormDiv = document.getElementById("dataset-importer-form");
  const datasetImporterBack = document.getElementById("dataset-importer-back");
  const dashFileInput = document.getElementById("dash-file-input");
  const dashProgress = document.getElementById("dash-progress");
  const dashProgressFill = document.getElementById("dash-progress-fill");
  const dashProgressText = document.getElementById("dash-progress-text");
  const dashProgressMessage = document.getElementById("dash-progress-message");
  const dashProgressEta = document.getElementById("dash-progress-eta");
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
      // Let user pick from existing favorite detectors
      const dets = favoriteDetectors || [];
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
          ? `/api/favorite-detectors/from-label-import/${encodeURIComponent(importer.name)}`
          : `/api/processor-importers/import/${encodeURIComponent(importer.name)}`;

        try {
          const res = await fetch(url, { method: "POST", headers, body });
          const result = await res.json();
          if (res.ok) {
            statusEl.textContent = `Created "${result.name || detName}"`;
            statusEl.style.color = "var(--color-good)";
            // Refresh detectors list so the new one is available
            try {
              const dRes = await fetch("/api/favorite-detectors");
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
    if (autopilotExamplesGrid) {
      renderExamplesGrid(autopilotExamplesGrid, model.examples || [], (updated) => {
        model.examples = updated;
        saveAutopilotExamples(model);
      });
    }
    // Populate autopilot example type dropdown with importer options (once)
    if (autopilotExampleType && !autopilotExampleType._importersLoaded) {
      autopilotExampleType._importersLoaded = true;
      Promise.all([
        fetch("/api/processor-importers").then(r => r.ok ? r.json() : []).catch(() => []),
        fetch("/api/label-importers").then(r => r.ok ? r.json() : []).catch(() => []),
      ]).then(([procImps, lblImps]) => {
        for (const imp of procImps) {
          const opt = document.createElement("option");
          opt.value = `proc_imp:${imp.name}`;
          opt.textContent = `${imp.icon || '\u{1F9E9}'} ${imp.display_name}`;
          autopilotExampleType.appendChild(opt);
        }
        for (const imp of lblImps) {
          const opt = document.createElement("option");
          opt.value = `label_imp:${imp.name}`;
          opt.textContent = `${imp.icon || '\uD83C\uDFF7\uFE0F'} ${imp.display_name}`;
          autopilotExampleType.appendChild(opt);
        }
      });
    }
  }

  async function saveAutopilotExamples(model) {
    try {
      const url = model.trainable
        ? `/api/trainable-models/${encodeURIComponent(model.name)}/examples`
        : `/api/favorite-detectors/${encodeURIComponent(model.name)}/examples`;
      await fetch(url, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ examples: model.examples || [] }),
      });
    } catch (_) { /* ignore */ }
  }

  if (autopilotExampleAdd) {
    autopilotExampleAdd.addEventListener("click", async () => {
      if (!_dashboardTrainMode || !_dashboardTrainMode.model) return;
      const model = _dashboardTrainMode.model;
      if (!model.examples) model.examples = [];
      const ex = await promptForExample(autopilotExampleType.value);
      if (ex) {
        model.examples.push(ex);
        refreshAutopilotExamples();
        saveAutopilotExamples(model);
      }
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
      hardLabels: 0, smartStatus: "", stableStatus: "",
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
        const res = await fetch("/api/favorite-detectors");
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
    if (menuLabelsImport) menuLabelsImport.classList.add("disabled");
    if (menuLabelsExport) menuLabelsExport.classList.add("disabled");
    if (menuDetectorImport) menuDetectorImport.classList.add("disabled");
    if (menuDetectorExport) menuDetectorExport.classList.add("disabled");
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
    if (menuLabelsImport) menuLabelsImport.classList.remove("disabled");
    if (menuDetectorImport) menuDetectorImport.classList.remove("disabled");
    // menuLabelsExport and menuDetectorExport stay disabled until votes are loaded (updateSortModeAvailability)
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

    // Show dashboard
    center.innerHTML = "";
    center.appendChild(dashboardView);
    center.className = "panel-center";
    dashboardView.style.display = "flex";
    dashProgress.style.display = "none";

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
      dashChangeDatasetBtn.style.display = "";
    } else {
      dashDatasetStatus.style.display = "none";
      dashChangeDatasetBtn.style.display = "none";
    }

    // Populate grids
    await renderDashboardDatasets();
    await renderDashboardModels();
    updateDashboardButtons();
  }

  function updateDashboardButtons() {
    const hasDataset = datasetLoaded || dashSelectedDataset;
    if (dashLabelBtn) dashLabelBtn.disabled = !hasDataset;
    if (dashDetectBtn) dashDetectBtn.disabled = !(hasDataset && dashSelectedDetector);
  }

  async function renderDashboardDatasets() {
    if (!dashDatasetGrid) return;

    if (datasetLoaded) {
      // Fetch dataset info and display it in the grid
      try {
        const res = await fetch("/api/dashboard/dataset-info");
        const info = await res.json();
        const mtInfo = mediaTypesMap[info.media_type];
        const icon = mtInfo ? mtInfo.icon : "";
        const typeName = mtInfo ? mtInfo.name : info.media_type || "media";
        const dupeSuffix = info.num_dupes ? ` (${info.num_dupes} dupes)` : "";
        dashDatasetGrid.innerHTML = `<table class="dash-dataset-table">
          <thead><tr>
            <th>Name</th>
            <th>Type</th>
            <th>Items</th>
            <th>Origin</th>
            <th class="col-actions-header"></th>
          </tr></thead>
          <tbody></tbody></table>`;
        const tr = document.createElement("tr");
        tr.className = "dash-dataset-row dash-selected";
        const nameTd = document.createElement("td");
        nameTd.className = "col-name";
        nameTd.innerHTML = `<span class="dash-name-text">${escapeHtml(info.name)}</span><button class="btn-icon dash-rename-btn" title="Rename" aria-label="Rename dataset">&#9998;</button>`;
        tr.appendChild(nameTd);
        tr.insertAdjacentHTML("beforeend", `
          <td class="col-type">${escapeHtml(icon)} ${escapeHtml(typeName)}</td>
          <td class="col-count">${info.num_medias}${escapeHtml(dupeSuffix)}</td>
          <td class="col-origin">${escapeHtml(info.origin)}</td>
          <td class="col-actions"><button class="btn-icon btn-icon-danger dash-delete-btn" title="Remove dataset" aria-label="Remove dataset">&#128465;</button></td>
        `);

        // Inline rename for dataset
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
                await fetch("/api/dashboard/dataset-rename", {
                  method: "PUT",
                  headers: { "Content-Type": "application/json" },
                  body: JSON.stringify({ name: newName }),
                });
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
          if (!confirm("Remove this dataset? All votes and labels will be cleared.")) return;
          try {
            await fetch("/api/dataset/clear", { method: "POST" });
            datasetLoaded = false;
            dashDatasetStatus.style.display = "none";
            dashChangeDatasetBtn.style.display = "none";
            await renderDashboardDatasets();
            updateDashboardButtons();
          } catch (_) {}
        });

        dashDatasetGrid.querySelector("tbody").appendChild(tr);
      } catch (_) {
        dashDatasetGrid.innerHTML = "";
      }
      return;
    }

    dashDatasetGrid.innerHTML = '<p style="color:var(--text-muted); padding:16px;">No dataset loaded yet. Use "+" to load one.</p>';
  }

  async function renderDashboardModels() {
    if (!dashModelGrid) return;

    // Fetch fresh favorites
    try {
      const res = await fetch("/api/favorite-detectors");
      const data = await res.json();
      favoriteDetectors = data.detectors || [];
    } catch (_) {}

    if (favoriteDetectors.length === 0) {
      dashModelGrid.innerHTML = '<p style="color:var(--text-muted); padding:16px;">No detectors loaded yet. Use "+" to create one.</p>';
      return;
    }

    const mediaIcons = Object.fromEntries(Object.entries(mediaTypesMap).map(([k, v]) => [k, v.icon]));

    let html = `<table class="dash-model-table">
      <thead><tr>
        <th data-sort="name">Name<span class="sort-arrow"></span></th>
        <th data-sort="media_type">Type<span class="sort-arrow"></span></th>
        <th>Examples</th>
        <th data-sort="num_labels" style="text-align:right">#TrainingLabels<span class="sort-arrow"></span></th>
        <th data-sort="autodetect" style="text-align:center">Fav<span class="sort-arrow"></span></th>
        <th data-sort="created_at">Created<span class="sort-arrow"></span></th>
        <th class="col-actions-header"></th>
      </tr></thead><tbody></tbody></table>`;
    dashModelGrid.innerHTML = html;

    const table = dashModelGrid.querySelector(".dash-model-table");
    let modelSort = { key: "name", asc: true };

    function renderModelRows() {
      const sorted = [...favoriteDetectors].sort((a, b) => {
        let va = a[modelSort.key], vb = b[modelSort.key];
        if (modelSort.key === "num_labels") return modelSort.asc ? (va || 0) - (vb || 0) : (vb || 0) - (va || 0);
        if (modelSort.key === "created_at") return modelSort.asc ? (va || 0) - (vb || 0) : (vb || 0) - (va || 0);
        if (modelSort.key === "autodetect") {
          va = va ? 1 : 0; vb = vb ? 1 : 0;
          return modelSort.asc ? va - vb : vb - va;
        }
        va = String(va || "").toLowerCase(); vb = String(vb || "").toLowerCase();
        return modelSort.asc ? va.localeCompare(vb) : vb.localeCompare(va);
      });

      const tbody = table.querySelector("tbody");
      tbody.innerHTML = "";
      sorted.forEach(det => {
        const icon = mediaIcons[det.media_type] || "\uD83D\uDD0D";
        const created = det.created_at ? new Date(det.created_at * 1000).toLocaleDateString() : "";
        const isSelected = dashSelectedDetector === det.name;
        const isFav = det.autodetect;
        const numLabels = det.num_labels || 0;
        const tr = document.createElement("tr");
        tr.className = "dash-model-row" + (isSelected ? " dash-selected" : "");
        tr.setAttribute("role", "button");
        tr.setAttribute("tabindex", "0");
        const exArr = det.examples || [];
        const exSummary = exArr.length === 0
          ? '<span style="color:var(--text-muted)">none</span>'
          : escapeHtml(exArr.map(e => e.value).join(", ")).substring(0, 40) + (exArr.map(e => e.value).join(", ").length > 40 ? "\u2026" : "");
        tr.innerHTML = `
          <td class="col-name"><span class="dash-name-text">${escapeHtml(det.name)}</span><button class="btn-icon dash-rename-btn" title="Rename" aria-label="Rename model">&#9998;</button></td>
          <td class="col-type">${escapeHtml(icon)} ${escapeHtml(det.media_type)}</td>
          <td class="col-examples"><span class="dash-examples-text">${exSummary}</span><button class="btn-icon dash-edit-examples-btn" title="Edit examples" aria-label="Edit examples">&#9998;</button></td>
          <td class="col-num-labels" style="text-align:right">${numLabels > 0 ? numLabels : '<span style="color:var(--text-muted)">0</span>'}</td>
          <td class="col-fav" style="text-align:center"><input type="checkbox" class="fav-checkbox" ${isFav ? "checked" : ""} aria-label="Favorite"></td>
          <td class="col-date">${escapeHtml(created)}</td>
          <td class="col-actions"><button class="btn-icon btn-icon-danger dash-delete-btn" title="Remove model" aria-label="Remove model">&#128465;</button></td>
        `;
        // Inline rename for model
        const renameBtn = tr.querySelector(".dash-rename-btn");
        renameBtn.addEventListener("click", (e) => {
          e.stopPropagation();
          const nameTd = tr.querySelector(".col-name");
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
                const res = await fetch(`/api/favorite-detectors/${encodeURIComponent(current)}/rename`, {
                  method: "PUT",
                  headers: { "Content-Type": "application/json" },
                  body: JSON.stringify({ new_name: newName }),
                });
                if (res.ok) {
                  det.name = newName;
                  if (dashSelectedDetector === current) dashSelectedDetector = newName;
                  nameSpan.textContent = newName;
                }
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
        // Edit examples button
        const editExBtn = tr.querySelector(".dash-edit-examples-btn");
        editExBtn.addEventListener("click", (e) => {
          e.stopPropagation();
          openExamplesEditorModal(det);
        });
        // Delete model
        const deleteBtn = tr.querySelector(".dash-delete-btn");
        deleteBtn.addEventListener("click", async (e) => {
          e.stopPropagation();
          if (!confirm(`Delete model "${det.name}"? This cannot be undone.`)) return;
          try {
            const res = await fetch(`/api/favorite-detectors/${encodeURIComponent(det.name)}`, { method: "DELETE" });
            if (res.ok) {
              favoriteDetectors = favoriteDetectors.filter(d => d.name !== det.name);
              if (dashSelectedDetector === det.name) dashSelectedDetector = null;
              renderModelRows();
              updateDashboardButtons();
              if (favoriteDetectors.length === 0) {
                dashModelGrid.innerHTML = '<p style="color:var(--text-muted); padding:16px;">No detectors loaded yet. Use "+" to create one.</p>';
              }
            }
          } catch (_) {}
        });
        // Favorite checkbox toggle
        const checkbox = tr.querySelector(".fav-checkbox");
        checkbox.addEventListener("click", async (e) => {
          e.stopPropagation();
          const newVal = checkbox.checked;
          try {
            await fetch(`/api/favorite-detectors/${encodeURIComponent(det.name)}/autodetect`, {
              method: "PUT",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ autodetect: newVal }),
            });
            det.autodetect = newVal;
          } catch (_) {
            checkbox.checked = !newVal; // revert on failure
          }
        });
        tr.addEventListener("click", () => {
          dashSelectedDetector = det.name;
          renderModelRows();
          updateDashboardButtons();
        });
        tr.addEventListener("keydown", (e) => {
          if (e.key === "Enter" || e.key === " ") { e.preventDefault(); tr.click(); }
        });
        tbody.appendChild(tr);
      });

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
  }

  // Dashboard: load a dataset from the selected demo, then run a callback
  async function dashLoadSelectedDataset(callback) {
    if (!dashSelectedDataset) return;

    dashProgress.style.display = "block";
    dashProgressFill.style.width = "0%";
    dashProgressFill.classList.add("indeterminate");
    dashProgressText.textContent = "";
    dashProgressMessage.textContent = "Loading...";
    dashProgressMessage.style.color = "var(--text-secondary)";
    dashProgressEta.textContent = "";

    try {
      const res = await fetch("/api/dataset/load-demo", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: dashSelectedDataset.name }),
      });
      if (!res.ok) {
        const error = await res.json();
        dashProgressMessage.textContent = `Error: ${error.error}`;
        dashProgressMessage.style.color = "var(--color-bad)";
        return;
      }
    } catch (e) {
      dashProgressMessage.textContent = `Error: ${e.message}`;
      dashProgressMessage.style.color = "var(--color-bad)";
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
        dashProgressMessage.textContent = `Error: ${progress.error}`;
        dashProgressMessage.style.color = "var(--color-bad)";
        dashProgressFill.classList.remove("indeterminate");
        return;
      }

      if (progress.pct != null) {
        dashProgressFill.classList.remove("indeterminate");
        dashProgressFill.style.width = `${progress.pct}%`;
        dashProgressText.textContent = `${progress.pct}%`;
      }
      if (progress.message) {
        dashProgressMessage.textContent = progress.message;
        dashProgressMessage.style.color = "var(--text-secondary)";
      }

      if (progress.status === "idle") {
        stopDashProgressPolling();
        dashProgress.style.display = "none";

        // Refresh dataset state
        await checkDatasetStatus();
        if (datasetLoaded) {
          await fetchMedias();
          await fetchVotes();

          const cb = dashPendingAction;
          dashPendingAction = null;
          if (typeof cb === "function") cb();
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
    const res = await fetch("/api/dataset/progress");
    const progress = await res.json();

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
          try {
            const infoRes = await fetch("/api/dashboard/dataset-info");
            if (infoRes.ok) {
              const info = await infoRes.json();
              dashboardDatasets.push({
                id: _dashboardNextId++,
                name: info.name,
                num_medias: info.num_medias,
                media_type: info.media_type,
                origin: info.origin,
                source: info.source || null,
              });
            }
          } catch (_) { /* ignore */ }
          // Clear the dataset from the backend
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

          // Set text sort mode and trigger sort with the model's text query
          sortMode = "text";
          textSortWrap.style.display = "";
          learnedSortWrap.style.display = "none";
          loadSortWrap.style.display = "none";
          // Update radio buttons to reflect text sort mode
          document.querySelectorAll('input[name="sort-mode"]').forEach(r => {
            r.checked = r.value === "text";
          });
          textSortInput.value = trainInfo.model.text_query || "";
          if (trainInfo.model.text_query) {
            fetchTextSort(trainInfo.model.text_query);
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
            tr.setAttribute("aria-label", `${ds.label}: ${ds.description}`);
            tr.onclick = () => loadDemo(ds.name);
            tr.addEventListener("keydown", (e) => {
              if (e.key === "Enter" || e.key === " ") { e.preventDefault(); loadDemo(ds.name); }
            });
            const descShort = ds.description.length > 60 ? ds.description.slice(0, 57) + "…" : ds.description;
            tr.innerHTML = `
              <td class="col-name">${escapeHtml(ds.label)}</td>
              <td class="col-num">${ds.num_files}</td>
              <td class="col-num">${ds.num_categories}</td>
              <td class="col-desc" title="${escapeHtml(ds.description)}">${escapeHtml(descShort)}</td>
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

        // Fetch the user's favorite media types to pick the initial tab
        let initialTab = availableTypes[0] || Object.keys(mediaTypesMap)[0] || "audio";
        try {
          const settingsRes = await fetch("/api/settings");
          if (settingsRes.ok) {
            const settingsData = await settingsRes.json();
            const favs = settingsData.favorite_media_types || [];
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

  backButton.addEventListener("click", () => {
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
      saveTrainableModelLabels();
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

  // Dataset change — clears current dataset and returns to dashboard
  if (menuDatasetChange && burgerDropdown) {
    menuDatasetChange.addEventListener("click", async () => {
      if (await vtConfirm("Changing the dataset will erase your current dataset. Continue?")) {
        fetch("/api/dataset/clear", { method: "POST" })
          .then(() => {
            medias = [];
            votes = { good: [], bad: [], click_times: {}, learned_scores: {} };
            selected = null;
            datasetLoaded = false;
            dashSelectedDataset = null;
            showDashboard();
            renderVotes();
            updateLabelCounts();
            closeBurgerMenu();
          });
      } else {
        closeBurgerMenu();
      }
    });
  }


  // Labels export – open modal
  async function openLabelExporterModal() {
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
          runLabelExport(exp);
        });
        el.addEventListener("keydown", (e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            labelExporterModal.classList.remove("show");
            runLabelExport(exp);
          }
        });
      });
    }

    labelExporterModal.classList.add("show");
  }

  async function runLabelExport(exp) {
    menuLabelsStatus.textContent = "";
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

  if (menuLabelsExport) {
    menuLabelsExport.addEventListener("click", async () => {
      if (menuLabelsExport.classList.contains("disabled")) return;
      closeBurgerMenu();
      await openLabelExporterModal();
    });
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

  // Detector export – open modal
  if (menuDetectorExport) {
    menuDetectorExport.addEventListener("click", async () => {
      if (menuDetectorExport.classList.contains("disabled")) return;
      closeBurgerMenu();
      await openDetectorExportModal();
    });
  }

  async function openDetectorExportModal() {
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

    browserBtn.addEventListener("click", () => { detectorExportModal.classList.remove("show"); runDetectorExportBrowser(); });
    browserBtn.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); detectorExportModal.classList.remove("show"); runDetectorExportBrowser(); }
    });
    serverBtn.addEventListener("click", () => { detectorExportModal.classList.remove("show"); runDetectorExportServer(); });
    serverBtn.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); detectorExportModal.classList.remove("show"); runDetectorExportServer(); }
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
    menuDetectorStatus.textContent = "";
    // Collect required field values via prompts
    const fieldValues = {};
    for (const field of exp.fields) {
      if (field.required) {
        const val = await vtPrompt(field.label + (field.description ? ` (${field.description})` : ""), field.default || "");
        if (val === null) {
          menuDetectorStatus.textContent = "Export cancelled";
          setTimeout(() => { menuDetectorStatus.textContent = ""; }, 2000);
          return;
        }
        fieldValues[field.key] = val;
      } else {
        fieldValues[field.key] = field.default || "";
      }
    }

    menuDetectorStatus.textContent = "Exporting labels\u2026";
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
        menuDetectorStatus.textContent = result.error || "Export failed";
        setTimeout(() => { menuDetectorStatus.textContent = ""; }, 3000);
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
      menuDetectorStatus.textContent = result.message || "Labels exported";
      setTimeout(() => { menuDetectorStatus.textContent = ""; }, 4000);
    } catch (e) {
      menuDetectorStatus.textContent = "Export failed";
      setTimeout(() => { menuDetectorStatus.textContent = ""; }, 3000);
    }
  }

  async function runDetectorExportBrowser() {
    menuDetectorStatus.textContent = "Exporting detector\u2026";
    const res = await fetch("/api/detector/export", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
    });
    if (!res.ok) {
      menuDetectorStatus.textContent = "Export failed";
      setTimeout(() => { menuDetectorStatus.textContent = ""; }, 3000);
      return;
    }
    const data = await res.json();
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "detector.json";
    a.click();
    URL.revokeObjectURL(url);
    menuDetectorStatus.textContent = "Detector exported (browser)";
    setTimeout(() => { menuDetectorStatus.textContent = ""; }, 3000);
  }

  async function runDetectorExportServer() {
    const name = await vtPrompt("Enter a name for the detector file (saved on server):");
    if (!name || !name.trim()) return;

    menuDetectorStatus.textContent = "Saving detector to server\u2026";

    const res = await fetch("/api/detector/export-server", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: name.trim() }),
    });

    if (res.status === 409) {
      const info = await res.json();
      const overwrite = await vtConfirm(
        `A detector file "${info.name}.json" already exists on the server.\n\nPath: ${info.path}\n\nOverwrite it?`
      );
      if (overwrite) {
        menuDetectorStatus.textContent = "Overwriting detector on server\u2026";
        const res2 = await fetch("/api/detector/export-server", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ name: name.trim(), overwrite: true }),
        });
        if (res2.ok) {
          const data2 = await res2.json();
          menuDetectorStatus.textContent = `Saved to server: ${data2.name}.json`;
          setTimeout(() => { menuDetectorStatus.textContent = ""; }, 4000);
        } else {
          const err = await res2.json().catch(() => ({}));
          menuDetectorStatus.textContent = err.error || "Server export failed";
          setTimeout(() => { menuDetectorStatus.textContent = ""; }, 3000);
        }
      } else {
        const newName = await vtPrompt("Enter a different name:");
        if (newName && newName.trim()) {
          menuDetectorStatus.textContent = "Saving detector to server\u2026";
          const res3 = await fetch("/api/detector/export-server", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ name: newName.trim() }),
          });
          if (res3.ok) {
            const data3 = await res3.json();
            menuDetectorStatus.textContent = `Saved to server: ${data3.name}.json`;
            setTimeout(() => { menuDetectorStatus.textContent = ""; }, 4000);
          } else {
            const err = await res3.json().catch(() => ({}));
            menuDetectorStatus.textContent = err.error || "Server export failed";
            setTimeout(() => { menuDetectorStatus.textContent = ""; }, 3000);
          }
        } else {
          menuDetectorStatus.textContent = "Export cancelled";
          setTimeout(() => { menuDetectorStatus.textContent = ""; }, 2000);
        }
      }
    } else if (res.ok) {
      const data = await res.json();
      menuDetectorStatus.textContent = `Saved to server: ${data.name}.json`;
      setTimeout(() => { menuDetectorStatus.textContent = ""; }, 4000);
    } else {
      const err = await res.json().catch(() => ({}));
      menuDetectorStatus.textContent = err.error || "Server export failed";
      setTimeout(() => { menuDetectorStatus.textContent = ""; }, 3000);
    }
  }

  function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML.replace(/'/g, '&#39;');
  }

  function formatOrigin(hit) {
    const origin = hit.origin;
    if (!origin) return "";
    if (origin.params) {
      const firstVal = Object.values(origin.params)[0];
      if (firstVal) return `${origin.importer}(${firstVal})`;
    }
    return origin.importer || "";
  }

  function displayAutodetectResults(data) {
    // Collect all Good hits across all detectors
    const allHits = [];
    for (const result of Object.values(data.results)) {
      for (const hit of (result.hits || [])) {
        allHits.push(hit);
      }
    }

    // Display summary
    autodetectSummary.innerHTML = `
      <p style="color: var(--text-primary);">
        <strong>Media Type:</strong> ${data.media_type} &nbsp;|&nbsp;
        <strong>Detectors Run:</strong> ${data.detectors_run} &nbsp;|&nbsp;
        <strong>Good Results:</strong> ${allHits.length}
      </p>
    `;

    // Display results as a table
    if (allHits.length === 0) {
      autodetectResults.innerHTML = '<p style="color: var(--text-muted);">No positive hits found.</p>';
    } else {
      let tableHtml = `<table class="results-table">`;
      tableHtml += `<thead><tr>`;
      tableHtml += `<th>Origin</th>`;
      tableHtml += `<th>Name</th>`;
      tableHtml += `<th>MD5</th>`;
      tableHtml += `<th>Filename</th>`;
      tableHtml += `</tr></thead><tbody>`;
      for (const hit of allHits) {
        const origin = escapeHtml(formatOrigin(hit));
        const name = escapeHtml(hit.origin_name || hit.filename || "");
        const md5 = escapeHtml(hit.md5 || "");
        const filename = escapeHtml(hit.filename || "");
        tableHtml += `<tr>`;
        tableHtml += `<td class="col-secondary">${origin}</td>`;
        tableHtml += `<td>${name}</td>`;
        tableHtml += `<td class="col-muted">${md5}</td>`;
        tableHtml += `<td class="col-secondary">${filename}</td>`;
        tableHtml += `</tr>`;
      }
      tableHtml += `</tbody></table>`;
      autodetectResults.innerHTML = tableHtml;
    }

    // Store results for copying
    window.autodetectResultsData = data;
    window.autodetectAllHits = allHits;

    // Reset export controls
    const goodRadio = document.querySelector('input[name="export-sides"][value="good"]');
    if (goodRadio) goodRadio.checked = true;
    if (fillFromSortCheckbox) fillFromSortCheckbox.checked = false;
    if (fillFromSortInfo) fillFromSortInfo.textContent = "";
    setExportStatus("", "var(--text-muted)");

    // Load exporters if not already loaded
    if (exportersList.length === 0) loadExportersList();

    // Show modal
    autodetectModal.classList.add("show");
  }

  if (autodetectModalClose) {
    autodetectModalClose.addEventListener("click", () => {
      autodetectModal.classList.remove("show");
    });
  }

  if (copyResultsBtn) {
    copyResultsBtn.addEventListener("click", () => {
      const allHits = window.autodetectAllHits;
      if (!allHits || allHits.length === 0) return;

      const columnSelect = document.getElementById("copy-column-select");
      const separatorSelect = document.getElementById("copy-separator-select");
      const column = columnSelect ? columnSelect.value : "origin+name";
      const sepKey = separatorSelect ? separatorSelect.value : "newline";

      const separatorMap = { ",": ",", "tab": "\t", "space": " ", "newline": "\n" };
      const sep = separatorMap[sepKey] || "\n";

      const values = allHits.map(hit => {
        const origin = formatOrigin(hit);
        const name = hit.origin_name || hit.filename || "";
        switch (column) {
          case "origin+name":
            return origin ? `${origin}  ${name}` : name;
          case "name":
            return name;
          case "md5":
            return hit.md5 || "";
          case "filename":
            return hit.filename || "";
          case "origin":
            return origin;
          default:
            return name;
        }
      });

      const text = values.join(sep);
      navigator.clipboard.writeText(text).then(() => {
        copyResultsBtn.textContent = "Copied!";
        setTimeout(() => {
          copyResultsBtn.textContent = "Copy To Clipboard";
        }, 2000);
      });
    });
  }

  // ---- Export section in auto-detect modal ----

  const exportExporterSelect = document.getElementById("export-exporter-select");
  const exportExporterFields = document.getElementById("export-exporter-fields");
  const exportRunBtn = document.getElementById("export-run-btn");
  const exportStatus = document.getElementById("export-status");
  const fillFromSortCheckbox = document.getElementById("fill-from-sort-checkbox");
  const fillFromSortInfo = document.getElementById("fill-from-sort-info");
  let exportersList = [];

  function getSelectedExportSides() {
    const checked = document.querySelector('input[name="export-sides"]:checked');
    return checked ? checked.value : "good";
  }

  async function loadExportersList() {
    try {
      const res = await fetch("/api/exporters");
      if (res.ok) {
        exportersList = await res.json();
        renderExporterDropdown();
      }
    } catch (_) {}
  }

  function renderExporterDropdown() {
    if (!exportExporterSelect) return;
    exportExporterSelect.innerHTML = "";
    for (const exp of exportersList) {
      const opt = document.createElement("option");
      opt.value = exp.name;
      opt.textContent = `${exp.icon} ${exp.display_name}`;
      exportExporterSelect.appendChild(opt);
    }
    renderExporterFields();
  }

  function renderExporterFields() {
    if (!exportExporterFields || !exportExporterSelect) return;
    const name = exportExporterSelect.value;
    const exp = exportersList.find(e => e.name === name);
    if (!exp || exp.fields.length === 0) {
      exportExporterFields.innerHTML = "";
      return;
    }
    let html = "";
    for (const field of exp.fields) {
      html += `<div style="margin-bottom:8px;">`;
      html += `<label class="form-label" style="margin-bottom:3px;font-size:0.8rem;">${escapeHtml(field.label)}${field.required ? " *" : ""}</label>`;
      if (field.field_type === "select") {
        html += `<select name="${escapeHtml(field.key)}" data-export-field class="form-input">`;
        for (const opt of field.options) {
          html += `<option value="${escapeHtml(opt)}"${opt === field.default ? " selected" : ""}>${escapeHtml(opt)}</option>`;
        }
        html += `</select>`;
      } else {
        const itype = field.field_type === "password" ? "password" : (field.field_type === "email" ? "email" : "text");
        const placeholder = escapeHtml(field.placeholder || field.description || "");
        html += `<input type="${itype}" name="${escapeHtml(field.key)}" value="${escapeHtml(field.default)}" placeholder="${placeholder}" data-export-field class="form-input" ${field.required ? "required" : ""}>`;
      }
      html += `</div>`;
    }
    exportExporterFields.innerHTML = html;
  }

  if (exportExporterSelect) {
    exportExporterSelect.addEventListener("change", renderExporterFields);
  }

  function buildFilteredResults(sides) {
    const data = window.autodetectResultsData;
    if (!data || !data.results) return data || {};
    const filtered = {};
    for (const [detName, detResult] of Object.entries(data.results)) {
      const entry = { ...detResult };
      if (sides === "good") {
        entry.hits = detResult.hits || [];
        delete entry.negative_hits;
      } else if (sides === "bad") {
        entry.hits = detResult.negative_hits || [];
        entry.total_hits = entry.hits.length;
        delete entry.negative_hits;
      } else {
        // both
        const good = (detResult.hits || []).map(h => ({ ...h, label: "good" }));
        const bad = (detResult.negative_hits || []).map(h => ({ ...h, label: "bad" }));
        entry.hits = [...good, ...bad];
        entry.total_hits = entry.hits.length;
        delete entry.negative_hits;
      }
      filtered[detName] = entry;
    }
    return { ...data, results: filtered };
  }

  function updateFillFromSortInfo() {
    if (!fillFromSortInfo) return;
    if (!fillFromSortCheckbox || !fillFromSortCheckbox.checked) {
      fillFromSortInfo.textContent = "";
      return;
    }
    if (!sortOrder || threshold === null) {
      fillFromSortInfo.textContent = "No sort results available. Run a sort first.";
      fillFromSortInfo.style.color = "var(--text-muted)";
      return;
    }
    const sides = getSelectedExportSides();
    const votedIds = new Set([...votes.good, ...votes.bad]);
    let goodCount = 0;
    let badCount = 0;
    for (const entry of sortOrder) {
      if (votedIds.has(entry.id)) continue;
      if (entry.score >= threshold) goodCount++;
      else badCount++;
    }
    let msg = "";
    if (sides === "good") msg = `${goodCount} unlabeled element${goodCount !== 1 ? "s" : ""} above threshold will be labeled Good.`;
    else if (sides === "bad") msg = `${badCount} unlabeled element${badCount !== 1 ? "s" : ""} below threshold will be labeled Bad.`;
    else msg = `${goodCount} Good + ${badCount} Bad unlabeled element${goodCount + badCount !== 1 ? "s" : ""} will be labeled.`;
    fillFromSortInfo.textContent = msg;
    fillFromSortInfo.style.color = "var(--accent)";
  }

  if (fillFromSortCheckbox) {
    fillFromSortCheckbox.addEventListener("change", updateFillFromSortInfo);
  }
  document.querySelectorAll('input[name="export-sides"]').forEach(radio => {
    radio.addEventListener("change", () => {
      updateFillFromSortInfo();
      updateAutodetectSummary();
    });
  });

  function updateAutodetectSummary() {
    const data = window.autodetectResultsData;
    if (!data || !autodetectSummary) return;
    const sides = getSelectedExportSides();
    let goodTotal = 0;
    let badTotal = 0;
    for (const result of Object.values(data.results)) {
      goodTotal += (result.hits || []).length;
      badTotal += (result.negative_hits || []).length;
    }
    let countText;
    if (sides === "good") countText = `<strong>Good Results:</strong> ${goodTotal}`;
    else if (sides === "bad") countText = `<strong>Bad Results:</strong> ${badTotal}`;
    else countText = `<strong>Good:</strong> ${goodTotal} &nbsp; <strong>Bad:</strong> ${badTotal}`;
    autodetectSummary.innerHTML = `
      <p style="color: var(--text-primary);">
        <strong>Media Type:</strong> ${data.media_type} &nbsp;|&nbsp;
        <strong>Detectors Run:</strong> ${data.detectors_run} &nbsp;|&nbsp;
        ${countText}
      </p>
    `;
  }

  function setExportStatus(msg, color) {
    if (exportStatus) {
      exportStatus.textContent = msg;
      exportStatus.style.color = color || "var(--text-muted)";
    }
  }

  if (exportRunBtn) {
    exportRunBtn.addEventListener("click", async () => {
      const sides = getSelectedExportSides();
      const useFill = fillFromSortCheckbox && fillFromSortCheckbox.checked;

      if (useFill) {
        // Fill from Sort mode
        if (!sortOrder || threshold === null) {
          await vtAlert("No sort results available. Run a sort first.", "warning");
          return;
        }

        // Dry run to get counts
        const dryRes = await fetch("/api/labels/fill-from-sort", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            sort_results: sortOrder,
            threshold: threshold,
            sides: sides,
            confirm: false,
          }),
        });
        if (!dryRes.ok) {
          setExportStatus("Failed to compute fill counts.", "var(--color-bad)");
          return;
        }
        const counts = await dryRes.json();
        const total = (counts.good_count || 0) + (counts.bad_count || 0);
        if (total === 0) {
          await vtAlert("No unlabeled elements to fill. All elements in the sort results are already labeled.", "info");
          return;
        }

        let desc;
        if (sides === "good") desc = `${counts.good_count} Good label${counts.good_count !== 1 ? "s" : ""}`;
        else if (sides === "bad") desc = `${counts.bad_count} Bad label${counts.bad_count !== 1 ? "s" : ""}`;
        else desc = `${counts.good_count} Good + ${counts.bad_count} Bad label${total !== 1 ? "s" : ""}`;

        const confirmed = await vtConfirm(`This will add ${desc} to the LabelSet and export. Continue?`);
        if (!confirmed) return;

        setExportStatus("Filling labels\u2026", "var(--text-muted)");
        const fillRes = await fetch("/api/labels/fill-from-sort", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            sort_results: sortOrder,
            threshold: threshold,
            sides: sides,
            confirm: true,
          }),
        });
        if (!fillRes.ok) {
          setExportStatus("Failed to fill labels.", "var(--color-bad)");
          return;
        }
        const fillData = await fillRes.json();
        const resultsForExport = fillData.results;

        // Now export using selected exporter
        await runExporterWithResults(resultsForExport);

        // Refresh votes
        try {
          const vRes = await fetch("/api/votes");
          if (vRes.ok) {
            const vData = await vRes.json();
            votes = vData;
            renderVotes();
          }
        } catch (_) {}
      } else {
        // Standard auto-detect export
        const filteredResults = buildFilteredResults(sides);
        await runExporterWithResults(filteredResults);
      }
    });
  }

  async function runExporterWithResults(results) {
    if (!exportExporterSelect) return;
    const exporterName = exportExporterSelect.value;
    if (!exporterName) {
      setExportStatus("Select an exporter.", "var(--text-muted)");
      return;
    }

    // Gather field values
    const fieldEls = exportExporterFields.querySelectorAll("[data-export-field]");
    const fieldValues = {};
    for (const el of fieldEls) {
      fieldValues[el.name] = el.value;
    }

    setExportStatus("Exporting\u2026", "var(--text-muted)");
    try {
      const res = await fetch("/api/exporters/export", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          exporter_name: exporterName,
          field_values: fieldValues,
          results: results,
        }),
      });
      const data = await res.json();
      if (res.ok && data.success) {
        setExportStatus(data.message || "Export complete.", "var(--color-good)");
      } else {
        setExportStatus(data.error || "Export failed.", "var(--color-bad)");
      }
    } catch (err) {
      setExportStatus(`Export error: ${err.message}`, "var(--color-bad)");
    }
  }

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
    const favList = data.favorite_media_types || [];
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

  // Favorite media type toggles
  favMtCheckboxes.forEach(cb => {
    cb.addEventListener("change", () => {
      const selected = [];
      favMtCheckboxes.forEach(c => { if (c.checked) selected.push(c.dataset.mediaType); });
      fetch("/api/settings", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ favorite_media_types: selected }),
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
        delete data.favorite_processors;
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

    // Disable Export Detector when no detector can be trained (need good AND bad)
    if (menuDetectorExport) {
      if (hasGoodAndBad) {
        menuDetectorExport.classList.remove("disabled");
      } else {
        menuDetectorExport.classList.add("disabled");
      }
    }

    // Disable Export Labels when no labels exist (need any votes)
    const hasAnyVotes = votes.good.length > 0 || votes.bad.length > 0;
    if (menuLabelsExport) {
      if (hasAnyVotes) {
        menuLabelsExport.classList.remove("disabled");
      } else {
        menuLabelsExport.classList.add("disabled");
      }
    }
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
      const res = await fetch("/api/diversity-tree/next");
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
      return !votes.good.includes(item.id) && !votes.bad.includes(item.id);
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
    medias = await res.json();
    renderMediaList();
  }

  async function fetchVotes() {
    const res = await fetch("/api/votes");
    votes = await res.json();
    renderVotes();
    renderStripe();
    updateSortModeAvailability();
    if (selected) renderCenter();
  }

  async function fetchInclusion() {
    const res = await fetch("/api/inclusion");
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
      const isGood = votes.good.includes(c.id);
      const isBad = votes.bad.includes(c.id);
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

  function renderCenter() {
    const c = medias.find(x => x.id === selected);
    if (!c) return;
    const isGood = votes.good.includes(c.id);
    const isBad = votes.bad.includes(c.id);
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
          <span class="metadata-value">${c.frequency} Hz</span>
        </div>` : ''}
        ${c.category && c.category !== 'unknown' ? `
        <div class="metadata-item">
          <span class="metadata-label">Category</span>
          <span class="metadata-value">${escapeHtml(c.category)}</span>
        </div>` : ''}
        <div class="metadata-item">
          <span class="metadata-label">Media Type</span>
          <span class="metadata-value">${mtInfo ? mtInfo.name : mediaType}</span>
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
        <div class="metadata-item">
          <span class="metadata-label">File Size</span>
          <span class="metadata-value">${(c.file_size / 1024).toFixed(1)} KB</span>
        </div>
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
        window.addEventListener("mousemove", (e) => {
          if (!isPanning) return;
          ivcPanX = panOriginX + (e.clientX - panStartX);
          ivcPanY = panOriginY + (e.clientY - panStartY);
          applyTransform();
          wrap.style.cursor = 'grabbing';
        });
        window.addEventListener("mouseup", () => {
          if (!isPanning) return;
          isPanning = false;
          const max = getMaxPan();
          wrap.style.cursor = (max.x > 0 || max.y > 0) ? 'grab' : '';
        });
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
      await fetch(`/api/medias/${id}/vote`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ vote }),
      });
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
        html += `<img class="vote-thumbnail" src="${thumbnailUrl(media)}" alt="${entry.name}" loading="lazy">`;
      }
      html += `<div class="vote-thumb-info">`;
    }
    html += `<span class="vote-name">${entry.name}</span><span class="vote-meta">${metaParts.join(" \u00b7 ")}</span>`;
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
      const isGood = votes.good.includes(item.id);
      const isBad = votes.bad.includes(item.id);
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
        const detRes = await fetch("/api/favorite-detectors");
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

  function showLabelImporterForm(importer) {
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
        const res = await fetch(`/api/favorite-detectors/from-label-import/${encodeURIComponent(importer.name)}`, {
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

  function renderErrorCostChart(errorCostData) {
    const canvas = document.getElementById("error-cost-chart");
    const ctx = canvas.getContext("2d");

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!errorCostData || errorCostData.length === 0) {
      ctx.fillStyle = themeColor("--text-muted");
      ctx.font = "14px sans-serif";
      ctx.fillText("No data available", 20, canvas.height / 2);
      return;
    }

    // Extract data
    const numLabels = errorCostData.map(d => d.num_labels);
    const errorCosts = errorCostData.map(d => d.error_cost);

    // Chart dimensions
    const padding = { top: 20, right: 20, bottom: 40, left: 50 };
    const chartWidth = canvas.width - padding.left - padding.right;
    const chartHeight = canvas.height - padding.top - padding.bottom;

    // Scales
    const maxLabels = Math.max(...numLabels);
    const maxCost = Math.max(...errorCosts);
    const minCost = Math.min(...errorCosts);

    const xScale = (val) => padding.left + (val / maxLabels) * chartWidth;
    const yScale = (val) => padding.top + chartHeight - ((val - minCost) / (maxCost - minCost || 1)) * chartHeight;

    // Draw axes
    ctx.strokeStyle = themeColor("--border");
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padding.left, padding.top);
    ctx.lineTo(padding.left, padding.top + chartHeight);
    ctx.lineTo(padding.left + chartWidth, padding.top + chartHeight);
    ctx.stroke();

    // Draw grid lines
    ctx.strokeStyle = themeColor("--border-subtle");
    ctx.lineWidth = 1;
    for (let i = 1; i <= 5; i++) {
      const y = padding.top + (chartHeight * i) / 5;
      ctx.beginPath();
      ctx.moveTo(padding.left, y);
      ctx.lineTo(padding.left + chartWidth, y);
      ctx.stroke();
    }

    // Draw line
    ctx.strokeStyle = themeColor("--accent");
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < errorCostData.length; i++) {
      const x = xScale(numLabels[i]);
      const y = yScale(errorCosts[i]);
      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    }
    ctx.stroke();

    // Draw points
    ctx.fillStyle = themeColor("--accent");
    for (let i = 0; i < errorCostData.length; i++) {
      const x = xScale(numLabels[i]);
      const y = yScale(errorCosts[i]);
      ctx.beginPath();
      ctx.arc(x, y, 3, 0, 2 * Math.PI);
      ctx.fill();
    }

    // Labels
    ctx.fillStyle = themeColor("--text-secondary");
    ctx.font = "12px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("Number of Labels", canvas.width / 2, canvas.height - 10);

    ctx.save();
    ctx.translate(15, canvas.height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("Error Cost", 0, 0);
    ctx.restore();

    // Axis labels
    ctx.textAlign = "center";
    ctx.fillText("0", padding.left, canvas.height - padding.bottom + 15);
    ctx.fillText(maxLabels.toString(), padding.left + chartWidth, canvas.height - padding.bottom + 15);

    ctx.textAlign = "right";
    ctx.fillText(maxCost.toFixed(2), padding.left - 5, padding.top + 5);
    ctx.fillText(minCost.toFixed(2), padding.left - 5, padding.top + chartHeight + 5);
  }

  function renderStabilityChart(stabilityData) {
    const canvas = document.getElementById("stability-chart");
    const ctx = canvas.getContext("2d");

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!stabilityData || stabilityData.length === 0) {
      ctx.fillStyle = themeColor("--text-muted");
      ctx.font = "14px sans-serif";
      ctx.fillText("No data available", 20, canvas.height / 2);
      return;
    }

    const dataToPlot = stabilityData;
    if (dataToPlot.length === 0) {
      ctx.fillStyle = themeColor("--text-muted");
      ctx.font = "14px sans-serif";
      ctx.fillText("Need more labels for stability analysis", 20, canvas.height / 2);
      return;
    }

    const numLabels = dataToPlot.map(d => d.num_labels);
    const numFlips = dataToPlot.map(d => d.num_flips);

    // Chart dimensions
    const padding = { top: 20, right: 20, bottom: 40, left: 50 };
    const chartWidth = canvas.width - padding.left - padding.right;
    const chartHeight = canvas.height - padding.top - padding.bottom;

    // Scales
    const maxLabels = Math.max(...numLabels);
    const maxFlips = Math.max(...numFlips, 1);

    const xScale = (val) => padding.left + (val / maxLabels) * chartWidth;
    const yScale = (val) => padding.top + chartHeight - (val / maxFlips) * chartHeight;

    // Draw axes
    ctx.strokeStyle = themeColor("--border");
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padding.left, padding.top);
    ctx.lineTo(padding.left, padding.top + chartHeight);
    ctx.lineTo(padding.left + chartWidth, padding.top + chartHeight);
    ctx.stroke();

    // Draw grid lines
    ctx.strokeStyle = themeColor("--border-subtle");
    ctx.lineWidth = 1;
    for (let i = 1; i <= 5; i++) {
      const y = padding.top + (chartHeight * i) / 5;
      ctx.beginPath();
      ctx.moveTo(padding.left, y);
      ctx.lineTo(padding.left + chartWidth, y);
      ctx.stroke();
    }

    // Draw line
    ctx.strokeStyle = themeColor("--color-good");
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < dataToPlot.length; i++) {
      const x = xScale(numLabels[i]);
      const y = yScale(numFlips[i]);
      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    }
    ctx.stroke();

    // Draw points
    ctx.fillStyle = themeColor("--color-good");
    for (let i = 0; i < dataToPlot.length; i++) {
      const x = xScale(numLabels[i]);
      const y = yScale(numFlips[i]);
      ctx.beginPath();
      ctx.arc(x, y, 3, 0, 2 * Math.PI);
      ctx.fill();
    }

    // Labels
    ctx.fillStyle = themeColor("--text-secondary");
    ctx.font = "12px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("Number of Labels", canvas.width / 2, canvas.height - 10);

    ctx.save();
    ctx.translate(15, canvas.height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("Prediction Flips", 0, 0);
    ctx.restore();

    // Axis labels
    ctx.textAlign = "center";
    ctx.fillText("0", padding.left, canvas.height - padding.bottom + 15);
    ctx.fillText(maxLabels.toString(), padding.left + chartWidth, canvas.height - padding.bottom + 15);

    ctx.textAlign = "right";
    ctx.fillText(maxFlips.toString(), padding.left - 5, padding.top + 5);
    ctx.fillText("0", padding.left - 5, padding.top + chartHeight + 5);
  }

  function renderDiversityChart(diversityData) {
    const canvas = document.getElementById("diversity-chart");
    const ctx = canvas.getContext("2d");

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!diversityData || diversityData.length === 0) {
      ctx.fillStyle = themeColor("--text-muted");
      ctx.font = "14px sans-serif";
      ctx.fillText("No data available", 20, canvas.height / 2);
      return;
    }

    const numLabels = diversityData.map(d => d.num_labels);
    const levels = diversityData.map(d => d.diversity_level);
    const treeDepth = diversityData[0].depth;

    const padding = { top: 20, right: 20, bottom: 40, left: 50 };
    const chartWidth = canvas.width - padding.left - padding.right;
    const chartHeight = canvas.height - padding.top - padding.bottom;

    const maxLabels = Math.max(...numLabels);
    const maxLevel = Math.max(treeDepth, Math.max(...levels), 1);
    const minLevel = Math.min(0, Math.min(...levels));

    const xScale = (val) => padding.left + (val / maxLabels) * chartWidth;
    const yScale = (val) => padding.top + chartHeight - ((val - minLevel) / (maxLevel - minLevel || 1)) * chartHeight;

    // Draw axes
    ctx.strokeStyle = themeColor("--border");
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padding.left, padding.top);
    ctx.lineTo(padding.left, padding.top + chartHeight);
    ctx.lineTo(padding.left + chartWidth, padding.top + chartHeight);
    ctx.stroke();

    // Draw grid lines
    ctx.strokeStyle = themeColor("--border-subtle");
    ctx.lineWidth = 1;
    for (let i = 1; i <= 5; i++) {
      const y = padding.top + (chartHeight * i) / 5;
      ctx.beginPath();
      ctx.moveTo(padding.left, y);
      ctx.lineTo(padding.left + chartWidth, y);
      ctx.stroke();
    }

    // Draw tree depth target line
    ctx.strokeStyle = themeColor("--color-good");
    ctx.lineWidth = 1;
    ctx.setLineDash([6, 4]);
    const depthY = yScale(treeDepth);
    ctx.beginPath();
    ctx.moveTo(padding.left, depthY);
    ctx.lineTo(padding.left + chartWidth, depthY);
    ctx.stroke();
    ctx.setLineDash([]);

    // Label the target line
    ctx.fillStyle = themeColor("--color-good");
    ctx.font = "11px sans-serif";
    ctx.textAlign = "left";
    ctx.fillText(`depth ${treeDepth}`, padding.left + chartWidth - 55, depthY - 5);

    // Draw line
    ctx.strokeStyle = themeColor("--accent");
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < diversityData.length; i++) {
      const x = xScale(numLabels[i]);
      const y = yScale(levels[i]);
      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    }
    ctx.stroke();

    // Draw points
    ctx.fillStyle = themeColor("--accent");
    for (let i = 0; i < diversityData.length; i++) {
      const x = xScale(numLabels[i]);
      const y = yScale(levels[i]);
      ctx.beginPath();
      ctx.arc(x, y, 3, 0, 2 * Math.PI);
      ctx.fill();
    }

    // Labels
    ctx.fillStyle = themeColor("--text-secondary");
    ctx.font = "12px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("Number of Labels", canvas.width / 2, canvas.height - 10);

    ctx.save();
    ctx.translate(15, canvas.height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("Diversity Level", 0, 0);
    ctx.restore();

    // Axis labels
    ctx.textAlign = "center";
    ctx.fillText("0", padding.left, canvas.height - padding.bottom + 15);
    ctx.fillText(maxLabels.toString(), padding.left + chartWidth, canvas.height - padding.bottom + 15);

    ctx.textAlign = "right";
    ctx.fillText(maxLevel.toFixed(1), padding.left - 5, padding.top + 5);
    ctx.fillText(minLevel.toFixed(1), padding.left - 5, padding.top + chartHeight + 5);
  }


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

      // Close modal and show dashboard progress
      datasetImporterModal.classList.remove("show");
      dashProgress.style.display = "block";
      dashProgressFill.style.width = "0%";
      dashProgressFill.classList.add("indeterminate");
      dashProgressText.textContent = "";
      dashProgressMessage.textContent = "Importing\u2026";
      dashProgressMessage.style.color = "var(--text-secondary)";
      dashProgressEta.textContent = "";

      try {
        const res = await fetch(`/api/dataset/import/${importer.name}`, { method: "POST", headers, body });
        if (!res.ok) {
          const err = await res.json();
          dashProgressMessage.textContent = `Error: ${err.error}`;
          dashProgressMessage.style.color = "var(--color-bad)";
          return;
        }
      } catch (err) {
        dashProgressMessage.textContent = `Error: ${err.message}`;
        dashProgressMessage.style.color = "var(--color-bad)";
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
        const favs = settingsData.favorite_media_types || [];
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

      dashProgress.style.display = "block";
      dashProgressFill.style.width = "0%";
      dashProgressFill.classList.add("indeterminate");
      dashProgressText.textContent = "";
      dashProgressMessage.textContent = "Uploading...";
      dashProgressMessage.style.color = "var(--text-secondary)";
      dashProgressEta.textContent = "";

      const formData = new FormData();
      formData.append("file", file);
      try {
        const res = await fetch("/api/dataset/load-file", { method: "POST", body: formData });
        if (!res.ok) {
          const data = await res.json();
          dashProgressMessage.textContent = `Error: ${data.error || "Upload failed"}`;
          dashProgressMessage.style.color = "var(--color-bad)";
          dashFileInput.value = "";
          return;
        }
      } catch (e) {
        dashProgressMessage.textContent = `Error: ${e.message}`;
        dashProgressMessage.style.color = "var(--color-bad)";
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

  // Dashboard: Change Dataset button
  if (dashChangeDatasetBtn) {
    dashChangeDatasetBtn.addEventListener("click", async () => {
      if (await vtConfirm("Changing the dataset will erase your current dataset. Continue?")) {
        await fetch("/api/dataset/clear", { method: "POST" });
        medias = [];
        votes = { good: [], bad: [], click_times: {}, learned_scores: {} };
        selected = null;
        datasetLoaded = false;
        dashSelectedDataset = null;
        showDashboard();
      }
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
          if (imp) showLabelImporterForm(imp);
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

    // Build media type options from the registry
    const mtOptions = Object.entries(mediaTypesMap).map(([id, mt]) =>
      `<option value="${escapeHtml(id)}">${escapeHtml(mt.icon || "")} ${escapeHtml(mt.name || id)}</option>`
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

    let html = `<h3 class="form-heading">New Model</h3>`;
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
        const res = await fetch("/api/favorite-detectors", {
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
        const url = `/api/favorite-detectors/${encodeURIComponent(det.name)}/examples`;
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
      // Activate train mode when a model is selected
      if (dashSelectedDetector) {
        const det = (favoriteDetectors || []).find(d => d.name === dashSelectedDetector);
        if (det) {
          _dashboardTrainMode = { model: det };
        }
      } else {
        _dashboardTrainMode = null;
      }

      if (datasetLoaded) {
        // Dataset already loaded — go straight to labeling
        showMainUI();
        if (medias.length > 0 && !selected) {
          selectMedia(medias[0].id);
        }
        return;
      }
      if (dashSelectedDataset) {
        // Need to load the selected demo dataset first, then go to labeling
        dashLoadSelectedDataset(() => {
          showMainUI();
          if (medias.length > 0 && !selected) {
            selectMedia(medias[0].id);
          }
        });
      }
    });
  }

  // Dashboard: Detect button
  if (dashDetectBtn) {
    dashDetectBtn.addEventListener("click", async () => {
      async function runDetect() {
        if (!dashSelectedDetector) {
          await vtAlert("Select a detector from the Model grid first.", "warning");
          return;
        }
        // Run auto-detect
        autodetectProgressModal.classList.add("show");
        autodetectProgressText.textContent = "Running auto-detect...";
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
            await vtAlert("Auto-detect failed. Make sure you have saved a detector for this media type.", "error");
            return;
          }
          res.json().then(data => displayAutodetectResults(data));
        }, 500);
      }

      if (datasetLoaded) {
        await runDetect();
      } else if (dashSelectedDataset) {
        dashLoadSelectedDataset(() => runDetect());
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
      [autodetectModal, autodetectModalClose],
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
    // Never intercept keys when a dialog is open
    if (vtDialogModal.classList.contains("show")) return;

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
