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
  let favoriteDetectors = [];  // List of favorite detectors
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
  // Media type metadata fetched from /api/media-types at startup.
  // Keyed by type_id → { type_id, name, icon, tab_title, loops, ... }
  let mediaTypesMap = {};
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

      vtDialogActions.innerHTML = "";
      buttons.forEach((btn) => {
        const el = document.createElement("button");
        el.className = "vt-dialog-btn " + (btn.primary ? "primary" : "secondary");
        el.textContent = btn.label;
        el.addEventListener("click", () => {
          vtDialogModal.classList.remove("show");
          resolve(btn.value === "input" ? vtDialogInput.value : btn.value);
        });
        vtDialogActions.appendChild(el);
      });

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
  const menuFavoritesStatus = document.getElementById("menu-favorites-status");
  const favPregenBtn = document.getElementById("fav-pregen-btn");
  const menuFavoritesManage = document.getElementById("menu-favorites-manage");
  const favoritesModal = document.getElementById("favorites-modal");
  const favoritesModalClose = document.getElementById("favorites-modal-close");
  const favoritesList = document.getElementById("favorites-list");
  const favAddName = document.getElementById("fav-add-name");
  const favAddStatus = document.getElementById("fav-add-status");
  const favAddFromVotesBtn = document.getElementById("fav-add-from-votes-btn");
  const favImporterButtonsDiv = document.getElementById("fav-importer-buttons");
  const autodetectModal = document.getElementById("autodetect-modal");
  const autodetectModalClose = document.getElementById("autodetect-modal-close");
  const autodetectSummary = document.getElementById("autodetect-summary");
  const autodetectResults = document.getElementById("autodetect-results");
  const copyResultsBtn = document.getElementById("copy-results-btn");
  const autodetectProgressModal = document.getElementById("autodetect-progress-modal");
  const autodetectProgressText = document.getElementById("autodetect-progress-text");
  const autodetectProgressBar = document.getElementById("autodetect-progress-bar");

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
    });
    tabAutopilot.addEventListener("click", () => {
      tabAutopilot.classList.add("active");
      tabAutopilot.setAttribute("aria-selected", "true");
      tabManual.classList.remove("active");
      tabManual.setAttribute("aria-selected", "false");
      tabPanelAutopilot.style.display = "";
      tabPanelManual.style.display = "none";
    });
  }

  // ---- Dataset Management ----

  async function checkDatasetStatus() {
    const res = await fetch("/api/dataset/status");
    const status = await res.json();
    datasetLoaded = status.loaded;

    if (datasetLoaded) {
      showMainUI();
      const mtInfo = mediaTypesMap[status.media_type];
      const dupeSuffix = status.num_dupes ? ` (${status.num_dupes} dupes)` : "";
      datasetInfo.textContent = mtInfo
        ? `${mtInfo.icon} ${status.num_medias} ${mtInfo.name.toLowerCase()} loaded${dupeSuffix}`
        : `${status.num_medias} medias loaded${dupeSuffix}`;
    } else {
      showWelcomeScreen();
    }

    return status;
  }

  function showWelcomeScreen() {
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
    sortBar.style.display = "none";
    datasetBar.style.display = "none";
    trainDatasetBar.style.display = "none";
    trainDetectorBar.style.display = "none";
    mediaList.innerHTML = "";
    leftPanel.style.display = "none";
    if (rightPanel) rightPanel.style.display = "none";
    stripeContainer.innerHTML = "";
    if (menuLabelsImport) menuLabelsImport.classList.add("disabled");
    if (menuLabelsExport) menuLabelsExport.classList.add("disabled");
    if (menuDetectorImport) menuDetectorImport.classList.add("disabled");
    if (menuDetectorExport) menuDetectorExport.classList.add("disabled");
  }

  function showMainUI() {
    datasetWelcome.style.display = "none";
    leftPanel.style.display = "";
    if (rightPanel) rightPanel.style.display = "";
    sortBar.style.display = "block";
    datasetBar.style.display = "flex";
    if (!selected) {
      center.className = "panel-center empty";
      center.innerHTML = '<p>Select a media from the left panel</p>';
      announce("Dataset loaded. Select a media from the left panel to begin.");
    }
    if (menuLabelsImport) menuLabelsImport.classList.remove("disabled");
    if (menuDetectorImport) menuDetectorImport.classList.remove("disabled");
    // menuLabelsExport and menuDetectorExport stay disabled until votes are loaded (updateSortModeAvailability)
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
      // instead of loading the dataset into the main UI.
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

        // Auto-select first media if none selected
        if (medias.length > 0 && !selected) {
          selectMedia(medias[0].id);
        }

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
    // Collapse any open submenus
    document.querySelectorAll(".burger-submenu.show").forEach(s => {
      s.classList.remove("show");
      const parent = s.previousElementSibling;
      if (parent) parent.setAttribute("aria-expanded", "false");
    });
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

  // ---- Favorite Detectors ----

  async function loadFavoriteDetectors() {
    const res = await fetch("/api/favorite-detectors");
    const data = await res.json();
    favoriteDetectors = data.detectors || [];
    updateFavoritesList();
  }

  function updateFavoritesList() {
    if (favoriteDetectors.length === 0) {
      favoritesList.innerHTML = '<p style="color:var(--text-muted);">No favorite detectors saved yet.</p>';
      return;
    }

    const mediaIcons = Object.fromEntries(Object.entries(mediaTypesMap).map(([k, v]) => [k, v.icon]));
    favoritesList.innerHTML = "";
    favoriteDetectors.forEach(detector => {
      const icon = mediaIcons[detector.media_type] || "🔍";
      const created = detector.created_at
        ? new Date(detector.created_at * 1000).toLocaleDateString()
        : "";
      const row = document.createElement("div");
      row.className = "fav-card";
      row.innerHTML = `
        <div class="fav-card-info">
          <div class="fav-card-name">${escapeHtml(detector.name)}</div>
          <div class="fav-card-meta">
            <span class="fav-badge">${escapeHtml(icon)} ${escapeHtml(detector.media_type)}</span>
            <span>threshold&nbsp;${detector.threshold.toFixed(2)}</span>
            ${created ? `<span>${escapeHtml(created)}</span>` : ""}
          </div>
        </div>
        <div class="fav-card-actions">
          <button class="fav-rename-btn btn-sm">Rename</button>
          <button class="fav-delete-btn btn-sm-danger">Delete</button>
        </div>`;
      row.querySelector(".fav-rename-btn").addEventListener("click", () => renameDetector(detector.name));
      row.querySelector(".fav-delete-btn").addEventListener("click", () => deleteDetector(detector.name));
      favoritesList.appendChild(row);
    });
  }

  function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML.replace(/'/g, '&#39;');
  }

  async function renameDetector(oldName) {
    const newName = await vtPrompt(`Rename detector "${oldName}" to:`, oldName);
    if (!newName || newName === oldName) return;

    const res = await fetch(`/api/favorite-detectors/${encodeURIComponent(oldName)}/rename`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ new_name: newName }),
    });

    if (res.ok) {
      await loadFavoriteDetectors();
    } else {
      await vtAlert("Failed to rename detector. Name may already exist.", "error");
    }
  }

  async function deleteDetector(name) {
    if (!await vtConfirm(`Are you sure you want to delete detector "${name}"?`)) return;

    const res = await fetch(`/api/favorite-detectors/${encodeURIComponent(name)}`, {
      method: "DELETE",
    });

    if (res.ok) {
      await loadFavoriteDetectors();
    } else {
      await vtAlert("Failed to delete detector.", "error");
    }
  }

  if (menuFavoritesManage) {
    menuFavoritesManage.addEventListener("click", async () => {
      await loadFavoriteDetectors();
      loadFavImporterButtons();
      // Pre-fill name input with most recent text-sort suggestion
      if (favAddName && !favAddName.value.trim()) {
        try {
          const sugRes = await fetch("/api/textsort-suggestions");
          const sugData = await sugRes.json();
          if (sugData.suggestions && sugData.suggestions.length > 0) {
            favAddName.value = sugData.suggestions[sugData.suggestions.length - 1];
          }
        } catch (_) {}
      }
      favoritesModal.classList.add("show");
      closeBurgerMenu();
    });
  }

  if (favoritesModalClose) {
    favoritesModalClose.addEventListener("click", () => {
      favoritesModal.classList.remove("show");
    });
  }

  // ---- Add Detector panel inside Manage Favorites modal ----

  function setFavAddStatus(msg, color) {
    if (favAddStatus) {
      favAddStatus.textContent = msg;
      favAddStatus.style.color = color || "var(--text-secondary)";
    }
  }

  // Add from current votes (train a new detector from labelled medias)
  if (favAddFromVotesBtn) {
    favAddFromVotesBtn.addEventListener("click", async () => {
      if (votes.good.length === 0 || votes.bad.length === 0) {
        setFavAddStatus("Need at least one good and one bad vote first.", "var(--color-bad)");
        return;
      }
      const name = favAddName ? favAddName.value.trim() : "";
      if (!name) {
        setFavAddStatus("Enter a name first.", "var(--color-bad)");
        return;
      }
      setFavAddStatus("Training detector\u2026", "var(--text-secondary)");

      const exportRes = await fetch("/api/detector/export", { method: "POST" });
      if (!exportRes.ok) {
        setFavAddStatus("Failed to train detector.", "var(--color-bad)");
        return;
      }
      const detectorData = await exportRes.json();
      const mediaType = medias.length > 0 ? medias[0].type : "audio";

      const saveRes = await fetch("/api/favorite-detectors", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name,
          media_type: mediaType,
          weights: detectorData.weights,
          threshold: detectorData.threshold,
        }),
      });

      if (saveRes.ok) {
        setFavAddStatus(`Detector \u201c${name}\u201d saved (${mediaType}).`, "var(--color-good)");
        if (favAddName) favAddName.value = "";
        await loadFavoriteDetectors();
      } else {
        setFavAddStatus("Failed to save detector.", "var(--color-bad)");
      }
    });
  }

  // ---- Dynamic importer buttons (processor importers + label importers) ----

  async function loadFavImporterButtons() {
    if (!favImporterButtonsDiv) return;
    favImporterButtonsDiv.innerHTML = "";

    const [procRes, labelRes] = await Promise.all([
      fetch("/api/processor-importers").catch(() => null),
      fetch("/api/label-importers").catch(() => null),
    ]);

    const procImporters = procRes && procRes.ok ? await procRes.json() : [];
    const labelImporters = labelRes && labelRes.ok ? await labelRes.json() : [];

    // Filter out processor importers that train from labelsets (label_file, csv_label_file)
    const detectorImporters = procImporters.filter((imp) => !imp.name.includes("label"));

    // Helper: add a section header + row of buttons
    function addSection(title, buttons) {
      if (buttons.length === 0) return;
      const header = document.createElement("div");
      header.className = "fav-section-header";
      header.textContent = title;
      favImporterButtonsDiv.appendChild(header);
      const row = document.createElement("div");
      row.className = "fav-btn-row";
      for (const btn of buttons) row.appendChild(btn);
      favImporterButtonsDiv.appendChild(row);
    }

    // Helper: create a file-picker button for a processor importer
    function makeProcFileButton(imp, fileField) {
      const btn = document.createElement("button");
      btn.textContent = `${imp.icon || "\u{1F9E9}"} ${imp.display_name}`;
      btn.className = "fav-import-btn";
      btn.addEventListener("click", () => {
        const input = document.createElement("input");
        input.type = "file";
        input.accept = fileField.accept || "";
        input.style.display = "none";
        document.body.appendChild(input);
        input.addEventListener("change", async () => {
          const file = input.files[0];
          if (!file) { input.remove(); return; }
          const defaultName = file.name.replace(/\.[^/.]+$/, "");
          const detectorName = (favAddName && favAddName.value.trim()) || defaultName;
          setFavAddStatus(`Importing from ${imp.display_name}\u2026`, "var(--text-secondary)");
          const formData = new FormData();
          formData.append("file", file);
          formData.append("name", detectorName);
          const res = await fetch(`/api/processor-importers/import/${imp.name}`, {
            method: "POST",
            body: formData,
          });
          if (res.ok) {
            const data = await res.json();
            const detail = data.loaded != null ? `, ${data.loaded} files` : "";
            setFavAddStatus(`Saved \u201c${data.name}\u201d (${data.media_type}${detail}).`, "var(--color-good)");
            if (favAddName) favAddName.value = "";
            await loadFavoriteDetectors();
          } else {
            const err = await res.json().catch(() => ({}));
            setFavAddStatus(`Error: ${err.error || "Import failed"}`, "var(--color-bad)");
          }
          input.remove();
        });
        input.click();
      });
      return btn;
    }

    // Helper: create a text-prompt button for a processor importer (server path)
    function makeProcTextButton(imp, textField) {
      const btn = document.createElement("button");
      btn.textContent = `${imp.icon || "\u{1F9E9}"} ${imp.display_name}`;
      btn.className = "fav-import-btn";
      btn.addEventListener("click", async () => {
        const value = await vtPrompt(`Enter ${textField.label}:`, textField.placeholder || "");
        if (!value) return;
        const defaultName = value.split("/").pop().replace(/\.[^/.]+$/, "");
        const detectorName = (favAddName && favAddName.value.trim()) || defaultName;
        setFavAddStatus(`Importing from ${imp.display_name}\u2026`, "var(--text-secondary)");
        const body = { name: detectorName };
        body[textField.key] = value;
        const res = await fetch(`/api/processor-importers/import/${imp.name}`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
        if (res.ok) {
          const data = await res.json();
          const detail = data.loaded != null ? `, ${data.loaded} files` : "";
          setFavAddStatus(`Saved \u201c${data.name}\u201d (${data.media_type}${detail}).`, "var(--color-good)");
          if (favAddName) favAddName.value = "";
          await loadFavoriteDetectors();
        } else {
          const err = await res.json().catch(() => ({}));
          setFavAddStatus(`Error: ${err.error || "Import failed"}`, "var(--color-bad)");
        }
      });
      return btn;
    }

    // Helper: create a file-picker button for a label importer (trains detector)
    function makeLabelFileButton(imp, fileField) {
      const btn = document.createElement("button");
      btn.textContent = `${imp.icon || "\u{1F3F7}\uFE0F"} ${imp.display_name}`;
      btn.className = "fav-import-btn";
      btn.addEventListener("click", () => {
        const input = document.createElement("input");
        input.type = "file";
        input.accept = fileField.accept || "";
        input.style.display = "none";
        document.body.appendChild(input);
        input.addEventListener("change", async () => {
          const file = input.files[0];
          if (!file) { input.remove(); return; }
          const defaultName = file.name.replace(/\.[^/.]+$/, "");
          const detectorName = (favAddName && favAddName.value.trim()) || defaultName;
          setFavAddStatus(`Training from labelset (${imp.display_name})\u2026`, "var(--text-secondary)");
          const formData = new FormData();
          formData.append("file", file);
          formData.append("name", detectorName);
          const res = await fetch(`/api/favorite-detectors/from-label-import/${imp.name}`, {
            method: "POST",
            body: formData,
          });
          if (res.ok) {
            const data = await res.json();
            const detail = data.loaded != null ? `, ${data.loaded} matched` : "";
            setFavAddStatus(`Trained \u201c${data.name}\u201d (${data.media_type}${detail}).`, "var(--color-good)");
            if (favAddName) favAddName.value = "";
            await loadFavoriteDetectors();
          } else {
            const err = await res.json().catch(() => ({}));
            setFavAddStatus(`Error: ${err.error || "Training failed"}`, "var(--color-bad)");
          }
          input.remove();
        });
        input.click();
      });
      return btn;
    }

    // Helper: create a text-prompt button for a label importer (server path, trains detector)
    function makeLabelTextButton(imp, textField) {
      const btn = document.createElement("button");
      btn.textContent = `${imp.icon || "\u{1F3F7}\uFE0F"} ${imp.display_name}`;
      btn.className = "fav-import-btn";
      btn.addEventListener("click", async () => {
        const value = await vtPrompt(`Enter ${textField.label}:`, textField.placeholder || "");
        if (!value) return;
        const defaultName = value.split("/").pop().replace(/\.[^/.]+$/, "");
        const detectorName = (favAddName && favAddName.value.trim()) || defaultName;
        setFavAddStatus(`Training from labelset (${imp.display_name})\u2026`, "var(--text-secondary)");
        const body = { name: detectorName };
        body[textField.key] = value;
        const res = await fetch(`/api/favorite-detectors/from-label-import/${imp.name}`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
        if (res.ok) {
          const data = await res.json();
          const detail = data.loaded != null ? `, ${data.loaded} matched` : "";
          setFavAddStatus(`Trained \u201c${data.name}\u201d (${data.media_type}${detail}).`, "var(--color-good)");
          if (favAddName) favAddName.value = "";
          await loadFavoriteDetectors();
        } else {
          const err = await res.json().catch(() => ({}));
          setFavAddStatus(`Error: ${err.error || "Training failed"}`, "var(--color-bad)");
        }
      });
      return btn;
    }

    // Build buttons for processor importers (load pre-trained detector)
    const procButtons = [];
    for (const imp of detectorImporters) {
      const fileField = imp.fields.find((f) => f.field_type === "file");
      const textField = imp.fields.find((f) => f.field_type === "text");
      if (fileField) {
        procButtons.push(makeProcFileButton(imp, fileField));
      } else if (textField) {
        procButtons.push(makeProcTextButton(imp, textField));
      }
    }

    // Build buttons for label importers (train detector from labelset)
    const labelButtons = [];
    for (const imp of labelImporters) {
      const fileField = imp.fields.find((f) => f.field_type === "file");
      const textField = imp.fields.find((f) => f.field_type === "text");
      if (fileField) {
        labelButtons.push(makeLabelFileButton(imp, fileField));
      } else if (textField) {
        labelButtons.push(makeLabelTextButton(imp, textField));
      }
    }

    addSection("Import Detector", procButtons);
    addSection("Train from Labelset", labelButtons);
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
        // Update main UI controls that live outside the modal
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

  // ---- Pregen processors (in favorites modal) ----

  if (favPregenBtn) {
    favPregenBtn.addEventListener("click", async () => {
      favPregenBtn.disabled = true;
      if (favAddStatus) {
        favAddStatus.textContent = "Adding pregen processors\u2026";
        favAddStatus.style.color = "var(--text-muted)";
      }
      try {
        const res = await fetch("/api/pregen-processors/add", { method: "POST" });
        const result = await res.json();
        if (res.ok && result.success) {
          if (favAddStatus) {
            favAddStatus.textContent = `Added ${result.added.length} pregen processor(s)`;
            favAddStatus.style.color = "var(--color-good)";
            setTimeout(() => { favAddStatus.textContent = ""; }, 3000);
          }
        } else {
          if (favAddStatus) {
            favAddStatus.textContent = result.error || "Failed to add pregen processors";
            favAddStatus.style.color = "var(--color-bad)";
            setTimeout(() => { favAddStatus.textContent = ""; }, 3000);
          }
        }
      } catch (err) {
        if (favAddStatus) {
          favAddStatus.textContent = `Error: ${err.message}`;
          favAddStatus.style.color = "var(--color-bad)";
          setTimeout(() => { favAddStatus.textContent = ""; }, 3000);
        }
      } finally {
        favPregenBtn.disabled = false;
      }
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
            if (menuFavoritesStatus) {
              menuFavoritesStatus.textContent = msg;
              setTimeout(() => { menuFavoritesStatus.textContent = ""; }, 3000);
            }
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

  // ---- Dashboard ----

  const dashboardView = document.getElementById("dashboard-view");
  const dashboardDatasetsTbody = document.getElementById("dashboard-datasets-tbody");
  const dashboardDatasetsTable = document.getElementById("dashboard-datasets-table");
  const dashboardDatasetsEmpty = document.getElementById("dashboard-datasets-empty");
  const dashboardModelsTbody = document.getElementById("dashboard-models-tbody");
  const dashboardModelsTable = document.getElementById("dashboard-models-table");
  const dashboardModelsEmpty = document.getElementById("dashboard-models-empty");
  const dashboardTrainBtn = document.getElementById("dashboard-train-btn");
  const dashboardRunBtn = document.getElementById("dashboard-run-btn");
  const dashboardDatasetAdd = document.getElementById("dashboard-dataset-add");
  const dashboardModelAdd = document.getElementById("dashboard-model-add");
  const menuDashboard = document.getElementById("menu-dashboard");

  let dashboardDatasets = [];
  let dashboardModels = [];
  let dashboardDatasetSort = { key: "name", asc: true };
  let dashboardModelSort = { key: "name", asc: true };
  let dashboardSelectedDatasets = {};  // id -> true
  let dashboardSelectedModels = {};    // id -> true
  let _dashboardAddDatasetMode = false;
  let _dashboardTrainMode = null;  // {model, dataset} when training a trainable model
  let _dashboardNextId = 1;

  async function _persistTrainableModelLabels() {
    if (!_dashboardTrainMode) return;
    const modelName = _dashboardTrainMode.model.name;
    try {
      const res = await fetch(`/api/trainable-models/${encodeURIComponent(modelName)}/labels`, {
        method: "POST",
      });
      if (res.ok) {
        const result = await res.json();
        // Update the dashboard model entry with new label count
        const model = dashboardModels.find(m => m.name === modelName && m.trainable);
        if (model) {
          model.num_labels = result.num_labels || 0;
        }
      }
    } catch (_) { /* ignore save errors */ }
  }

  async function saveTrainableModelLabels() {
    await _persistTrainableModelLabels();
    _dashboardTrainMode = null;
  }

  async function loadTrainableModelsIntoDashboard() {
    try {
      const res = await fetch("/api/trainable-models");
      if (!res.ok) return;
      const data = await res.json();
      const serverModels = data.models || [];
      for (const sm of serverModels) {
        // Skip if already present in dashboardModels (match by name + trainable flag)
        if (dashboardModels.some(m => m.trainable && m.name === sm.name)) continue;
        dashboardModels.push({
          id: _dashboardNextId++,
          name: sm.name,
          num_labels: sm.num_labels || 0,
          media_type: "any",
          text_examples: sm.text_query || "-",
          media_examples: "-",
          origin: "Train New",
          trainable: true,
          text_query: sm.text_query || "",
        });
      }
    } catch (_) { /* ignore */ }
  }

  function showDashboard() {
    // If leaving a training session, save labels first
    if (_dashboardTrainMode) {
      saveTrainableModelLabels();
    }
    // Hide left/right panels and welcome screen
    leftPanel.style.display = "none";
    if (rightPanel) rightPanel.style.display = "none";
    sortBar.style.display = "none";
    datasetBar.style.display = "none";
    datasetWelcome.style.display = "none";
    trainDatasetBar.style.display = "none";
    trainDetectorBar.style.display = "none";
    center.className = "panel-center";
    center.innerHTML = "";
    center.appendChild(dashboardView);
    dashboardView.style.display = "flex";
    // Load trainable models from server, then render
    loadTrainableModelsIntoDashboard().then(() => {
      renderDashboardDatasets();
      renderDashboardModels();
      updateDashboardButtons();
    });
  }

  function hideDashboard() {
    dashboardView.style.display = "none";
  }

  // Sort helper
  function dashboardSortRows(rows, sortState) {
    const { key, asc } = sortState;
    return rows.slice().sort((a, b) => {
      let va = a[key], vb = b[key];
      if (typeof va === "number" && typeof vb === "number") {
        return asc ? va - vb : vb - va;
      }
      va = String(va || "").toLowerCase();
      vb = String(vb || "").toLowerCase();
      if (va < vb) return asc ? -1 : 1;
      if (va > vb) return asc ? 1 : -1;
      return 0;
    });
  }

  function renderDashboardDatasets() {
    if (dashboardDatasets.length === 0) {
      dashboardDatasetsTable.style.display = "none";
      dashboardDatasetsEmpty.style.display = "";
      return;
    }
    dashboardDatasetsEmpty.style.display = "none";
    dashboardDatasetsTable.style.display = "";

    // Update sort arrows
    dashboardDatasetsTable.querySelectorAll("th[data-col]").forEach(th => {
      const arrow = th.querySelector(".sort-arrow");
      arrow.textContent = th.dataset.col === dashboardDatasetSort.key
        ? (dashboardDatasetSort.asc ? " \u25B2" : " \u25BC") : "";
    });

    const sorted = dashboardSortRows(dashboardDatasets, dashboardDatasetSort);
    dashboardDatasetsTbody.innerHTML = sorted.map(ds => {
      const sel = dashboardSelectedDatasets[ds.id] ? " selected" : "";
      return `<tr data-id="${ds.id}" class="${sel}">
        <td title="${escapeHtml(ds.name)}">${escapeHtml(ds.name)} <button class="btn-icon" data-action="rename" data-id="${ds.id}" title="Rename">&#9998;</button></td>
        <td>${ds.num_medias}</td>
        <td>${escapeHtml(ds.media_type)}</td>
        <td title="${escapeHtml(ds.origin)}">${escapeHtml(ds.origin)}</td>
        <td><button class="btn-icon btn-icon-danger" data-action="remove" data-id="${ds.id}" title="Remove">&#128465;</button></td>
      </tr>`;
    }).join("");

    // Wire row click for selection
    dashboardDatasetsTbody.querySelectorAll("tr").forEach(tr => {
      tr.addEventListener("click", (e) => {
        if (e.target.closest("button")) return;
        const id = parseInt(tr.dataset.id);
        if (dashboardSelectedDatasets[id]) {
          delete dashboardSelectedDatasets[id];
        } else {
          dashboardSelectedDatasets[id] = true;
        }
        renderDashboardDatasets();
        updateDashboardButtons();
      });
    });

    // Wire action buttons
    dashboardDatasetsTbody.querySelectorAll("button[data-action]").forEach(btn => {
      btn.addEventListener("click", async () => {
        const id = parseInt(btn.dataset.id);
        if (btn.dataset.action === "remove") {
          dashboardDatasets = dashboardDatasets.filter(d => d.id !== id);
          delete dashboardSelectedDatasets[id];
          renderDashboardDatasets();
          updateDashboardButtons();
        } else if (btn.dataset.action === "rename") {
          const ds = dashboardDatasets.find(d => d.id === id);
          if (!ds) return;
          const newName = await vtPrompt(`Rename dataset "${ds.name}" to:`, ds.name);
          if (newName && newName !== ds.name) {
            ds.name = newName;
            renderDashboardDatasets();
          }
        }
      });
    });
  }

  function renderDashboardModels() {
    if (dashboardModels.length === 0) {
      dashboardModelsTable.style.display = "none";
      dashboardModelsEmpty.style.display = "";
      return;
    }
    dashboardModelsEmpty.style.display = "none";
    dashboardModelsTable.style.display = "";

    // Update sort arrows
    dashboardModelsTable.querySelectorAll("th[data-col]").forEach(th => {
      const arrow = th.querySelector(".sort-arrow");
      arrow.textContent = th.dataset.col === dashboardModelSort.key
        ? (dashboardModelSort.asc ? " \u25B2" : " \u25BC") : "";
    });

    const sorted = dashboardSortRows(dashboardModels, dashboardModelSort);
    dashboardModelsTbody.innerHTML = sorted.map(m => {
      const sel = dashboardSelectedModels[m.id] ? " selected" : "";
      const trainBadge = m.trainable ? ' <span class="trainable-badge">trainable</span>' : "";
      return `<tr data-id="${m.id}" class="${sel}">
        <td title="${escapeHtml(m.name)}">${escapeHtml(m.name)}${trainBadge} <button class="btn-icon" data-action="rename" data-id="${m.id}" title="Rename">&#9998;</button></td>
        <td>${m.num_labels}</td>
        <td>${escapeHtml(m.media_type)}</td>
        <td title="${escapeHtml(m.text_examples)}">${escapeHtml(m.text_examples)}</td>
        <td title="${escapeHtml(m.media_examples)}">${escapeHtml(m.media_examples)}</td>
        <td title="${escapeHtml(m.origin)}">${escapeHtml(m.origin)}</td>
        <td><button class="btn-icon btn-icon-danger" data-action="remove" data-id="${m.id}" title="Remove">&#128465;</button></td>
      </tr>`;
    }).join("");

    // Wire row click for selection
    dashboardModelsTbody.querySelectorAll("tr").forEach(tr => {
      tr.addEventListener("click", (e) => {
        if (e.target.closest("button")) return;
        const id = parseInt(tr.dataset.id);
        if (dashboardSelectedModels[id]) {
          delete dashboardSelectedModels[id];
        } else {
          dashboardSelectedModels[id] = true;
        }
        renderDashboardModels();
        updateDashboardButtons();
      });
    });

    // Wire action buttons
    dashboardModelsTbody.querySelectorAll("button[data-action]").forEach(btn => {
      btn.addEventListener("click", async () => {
        const id = parseInt(btn.dataset.id);
        if (btn.dataset.action === "remove") {
          const m = dashboardModels.find(m => m.id === id);
          if (m && m.trainable) {
            // Also delete from backend
            fetch(`/api/trainable-models/${encodeURIComponent(m.name)}`, { method: "DELETE" }).catch(() => {});
          }
          dashboardModels = dashboardModels.filter(m => m.id !== id);
          delete dashboardSelectedModels[id];
          renderDashboardModels();
          updateDashboardButtons();
        } else if (btn.dataset.action === "rename") {
          const m = dashboardModels.find(m => m.id === id);
          if (!m) return;
          const newName = await vtPrompt(`Rename model "${m.name}" to:`, m.name);
          if (newName && newName !== m.name) {
            if (m.trainable) {
              // Rename on backend too
              try {
                await fetch(`/api/trainable-models/${encodeURIComponent(m.name)}/rename`, {
                  method: "PUT",
                  headers: { "Content-Type": "application/json" },
                  body: JSON.stringify({ new_name: newName }),
                });
              } catch (_) { /* ignore */ }
            }
            m.name = newName;
            renderDashboardModels();
          }
        }
      });
    });
  }

  function updateDashboardButtons() {
    const numDS = Object.keys(dashboardSelectedDatasets).length;
    const numMD = Object.keys(dashboardSelectedModels).length;
    // Train: exactly 1 dataset + exactly 1 model (model must be trainable)
    let trainEnabled = numDS === 1 && numMD === 1;
    if (trainEnabled) {
      const modelId = parseInt(Object.keys(dashboardSelectedModels)[0]);
      const model = dashboardModels.find(m => m.id === modelId);
      trainEnabled = model && model.trainable;
    }
    dashboardTrainBtn.disabled = !trainEnabled;
    // Run: at least 1 dataset + at least 1 model
    dashboardRunBtn.disabled = !(numDS >= 1 && numMD >= 1);
  }

  // -- Train button: load dataset + import labels + enter labeling UI --
  if (dashboardTrainBtn) {
    dashboardTrainBtn.addEventListener("click", async () => {
      const dsId = parseInt(Object.keys(dashboardSelectedDatasets)[0]);
      const mdId = parseInt(Object.keys(dashboardSelectedModels)[0]);
      const dataset = dashboardDatasets.find(d => d.id === dsId);
      const model = dashboardModels.find(m => m.id === mdId);
      if (!dataset || !model || !model.trainable) return;

      if (!dataset.source) {
        await vtAlert("Cannot reload this dataset — no source info stored. Please re-add the dataset.", "warning");
        return;
      }

      // Enter training mode
      _dashboardTrainMode = { model, dataset };
      hideDashboard();
      // Show dataset/detector context bars
      trainDatasetName.textContent = dataset.name;
      trainDatasetBar.style.display = "";
      trainDetectorName.textContent = model.name;
      trainDetectorBar.style.display = "";

      // Kick off dataset reload from the stored source
      try {
        const res = await fetch("/api/dataset/load-source", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ source: dataset.source }),
        });
        if (!res.ok) {
          const err = await res.json();
          _dashboardTrainMode = null;
          await vtAlert(err.error || "Failed to reload dataset", "warning");
          showDashboard();
          return;
        }
      } catch (err) {
        _dashboardTrainMode = null;
        await vtAlert(`Failed to reload dataset: ${err.message}`, "warning");
        showDashboard();
        return;
      }

      startProgressPolling();
    });
  }

  // Wire sortable headers for datasets table
  if (dashboardDatasetsTable) {
    dashboardDatasetsTable.querySelectorAll("th[data-col]").forEach(th => {
      th.addEventListener("click", () => {
        const col = th.dataset.col;
        if (dashboardDatasetSort.key === col) {
          dashboardDatasetSort.asc = !dashboardDatasetSort.asc;
        } else {
          dashboardDatasetSort = { key: col, asc: true };
        }
        renderDashboardDatasets();
      });
    });
  }

  // Wire sortable headers for models table
  if (dashboardModelsTable) {
    dashboardModelsTable.querySelectorAll("th[data-col]").forEach(th => {
      th.addEventListener("click", () => {
        const col = th.dataset.col;
        if (dashboardModelSort.key === col) {
          dashboardModelSort.asc = !dashboardModelSort.asc;
        } else {
          dashboardModelSort = { key: col, asc: true };
        }
        renderDashboardModels();
      });
    });
  }

  // Dashboard "Add Dataset" — enters the welcome screen in dashboard mode
  if (dashboardDatasetAdd) {
    dashboardDatasetAdd.addEventListener("click", () => {
      _dashboardAddDatasetMode = true;
      hideDashboard();
      // Clear any existing dataset so we can load a fresh one
      fetch("/api/dataset/clear", { method: "POST" }).then(() => {
        medias = [];
        votes = { good: [], bad: [], click_times: {}, learned_scores: {} };
        selected = null;
        datasetLoaded = false;
        showWelcomeScreen();
      });
    });
  }

  // Dashboard "Add Model" — opens the processor importer modal in dashboard mode
  if (dashboardModelAdd) {
    dashboardModelAdd.addEventListener("click", () => {
      openProcessorImporterModalForDashboard();
    });
  }

  async function openProcessorImporterModalForDashboard() {
    let importers = [];
    try {
      const res = await fetch("/api/processor-importers");
      if (res.ok) importers = await res.json();
    } catch (_) { /* ignore */ }

    processorImporterFormDiv.style.display = "none";
    processorImporterFormDiv.innerHTML = "";
    processorImporterBack.style.display = "none";
    processorImporterList.style.display = "";

    // Build option cards: "Train New" first, then processor importers
    let html = `
      <div class="processor-importer-option option-card" data-name="__train_new__">
        <span class="option-card-icon">\u{1F9E0}</span>
        <div>
          <div class="option-card-title">Train New</div>
          <div class="option-card-desc">Create a trainable model with a text description. Add labels over time.</div>
        </div>
      </div>
    `;

    if (importers.length > 0) {
      html += importers.map(imp => `
        <div class="processor-importer-option option-card" data-name="${escapeHtml(imp.name)}">
          <span class="option-card-icon">${escapeHtml(imp.icon || '\u{1F9E9}')}</span>
          <div>
            <div class="option-card-title">${escapeHtml(imp.display_name)}</div>
            <div class="option-card-desc">${escapeHtml(imp.description)}</div>
          </div>
        </div>
      `).join("");
    }

    processorImporterList.innerHTML = html;

    processorImporterList.querySelectorAll(".processor-importer-option").forEach(el => {
      el.setAttribute("role", "button");
      el.setAttribute("tabindex", "0");
      const name = el.dataset.name;
      if (name === "__train_new__") {
        el.addEventListener("click", () => showTrainNewFormForDashboard());
        el.addEventListener("keydown", (e) => {
          if (e.key === "Enter" || e.key === " ") { e.preventDefault(); showTrainNewFormForDashboard(); }
        });
      } else {
        const imp = importers.find(i => i.name === name);
        el.addEventListener("click", () => showProcessorImporterFormForDashboard(imp));
        el.addEventListener("keydown", (e) => {
          if (e.key === "Enter" || e.key === " ") { e.preventDefault(); showProcessorImporterFormForDashboard(imp); }
        });
      }
    });

    processorImporterModal.classList.add("show");
  }

  function showTrainNewFormForDashboard() {
    processorImporterList.style.display = "none";
    processorImporterBack.style.display = "inline-block";

    let html = `<h3 class="form-heading">Train New Model</h3>`;
    html += `<form id="train-new-form">`;
    html += `<div class="form-group">`;
    html += `<label class="form-label">Model Name *</label>`;
    html += `<input type="text" name="name" placeholder="e.g. Dog Barks" class="form-input" required>`;
    html += `<div class="form-hint">A name for this trainable model.</div>`;
    html += `</div>`;
    html += `<div class="form-group">`;
    html += `<label class="form-label">Text Sort Query *</label>`;
    html += `<input type="text" name="text_query" placeholder="e.g. sounds of dogs barking" class="form-input" required>`;
    html += `<div class="form-hint">Describes what to look for. Used for initial text-based sorting.</div>`;
    html += `</div>`;
    html += `<div id="train-new-status" class="status-text compact"></div>`;
    html += `<button type="submit" class="btn-block-primary">Create Model</button>`;
    html += `</form>`;

    processorImporterFormDiv.innerHTML = html;
    processorImporterFormDiv.style.display = "block";

    const statusEl = processorImporterFormDiv.querySelector("#train-new-status");

    processorImporterFormDiv.querySelector("#train-new-form").addEventListener("submit", async (e) => {
      e.preventDefault();
      const formEl = e.target;
      const name = formEl.elements["name"].value.trim();
      const textQuery = formEl.elements["text_query"].value.trim();
      if (!name || !textQuery) return;

      statusEl.textContent = "Creating\u2026";
      statusEl.style.color = "var(--text-muted)";

      try {
        const res = await fetch("/api/trainable-models", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ name, text_query: textQuery }),
        });
        const result = await res.json();
        if (res.ok) {
          dashboardModels.push({
            id: _dashboardNextId++,
            name: result.name,
            num_labels: 0,
            media_type: "any",
            text_examples: result.text_query,
            media_examples: "-",
            origin: "Train New",
            trainable: true,
            text_query: result.text_query,
          });
          statusEl.textContent = `Created "${result.name}"`;
          statusEl.style.color = "var(--color-good)";
          setTimeout(() => {
            processorImporterModal.classList.remove("show");
            showDashboard();
          }, 800);
        } else {
          statusEl.textContent = result.error || "Creation failed";
          statusEl.style.color = "var(--color-bad)";
        }
      } catch (err) {
        statusEl.textContent = `Error: ${err.message}`;
        statusEl.style.color = "var(--color-bad)";
      }
    });
  }

  function showProcessorImporterFormForDashboard(importer) {
    processorImporterList.style.display = "none";
    processorImporterBack.style.display = "inline-block";

    let html = `<h3 class="form-heading">${escapeHtml(importer.display_name)}</h3>`;
    html += `<form id="proc-imp-form">`;
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
          const modelEntry = {
            id: _dashboardNextId++,
            name: result.name,
            num_labels: result.loaded || 0,
            media_type: result.media_type || "unknown",
            text_examples: "-",
            media_examples: "-",
            origin: importer.display_name,
          };
          dashboardModels.push(modelEntry);
          statusEl.textContent = `Imported "${result.name}"`;
          statusEl.style.color = "var(--color-good)";
          setTimeout(() => {
            processorImporterModal.classList.remove("show");
            showDashboard();
          }, 1000);
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

  // Burger menu handler for Dashboard
  if (menuDashboard) {
    menuDashboard.addEventListener("click", () => {
      closeBurgerMenu();
      showDashboard();
    });
  }

  // Initialize
  fetchMediaTypes().then(() => checkDatasetStatus()).then(async () => {
    if (datasetLoaded) {
      await fetchMedias();
      await fetchVotes();
      if (medias.length > 0 && !selected) {
        selectMedia(medias[0].id);
      }
    }
  });
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
  loadFavoriteDetectors();
  fetchLabelingStatus();
  loadSettings();

  // ---- Modal Escape key handler ----
  document.addEventListener("keydown", (e) => {
    if (e.key !== "Escape") return;

    // VT dialog gets priority (it's an alertdialog)
    if (vtDialogModal.classList.contains("show")) return;

    // Close any open modal on Escape, from most specific to least
    const modalClosePairs = [
      [labelImporterModal, labelImporterModalClose],
      [labelExporterModal, labelExporterModalClose],
      [detectorExportModal, detectorExportModalClose],
      [processorImporterModal, processorImporterModalClose],
      [favoritesModal, favoritesModalClose],
      [autodetectModal, autodetectModalClose],
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
