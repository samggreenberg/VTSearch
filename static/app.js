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
  const loadDetectorBtn = document.getElementById("load-detector-btn");
  const loadDetectorFile = document.getElementById("load-detector-file");
  const learnedRadio = document.getElementById("learned-radio");
  const loadRadio = document.getElementById("load-radio");
  const sortStatus = document.getElementById("sort-status");
  const sortProgress = document.getElementById("sort-progress");
  const sortProgressLabel = document.getElementById("sort-progress-label");
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
    sortStatus.textContent = "";
    sortProgressLabel.textContent = label;
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
          sortProgressLabel.textContent = `${pct}% — ${formatETA(remaining)}`;
          return;
        }
      }
      if (progress.message) {
        sortProgressLabel.textContent = progress.message;
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

  // Burger menu elements
  const burgerBtn = document.getElementById("burger-btn");
  const burgerDropdown = document.getElementById("burger-dropdown");
  const menuDatasetExport = document.getElementById("menu-dataset-export");
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
  const menuFavoritesStatus = document.getElementById("menu-favorites-status");
  const favPregenBtn = document.getElementById("fav-pregen-btn");
  const menuFavoritesManage = document.getElementById("menu-favorites-manage");
  const menuFavoritesAutodetect = document.getElementById("menu-favorites-autodetect");
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

  // ---- Dataset Management ----

  async function checkDatasetStatus() {
    const res = await fetch("/api/dataset/status");
    const status = await res.json();
    datasetLoaded = status.loaded;

    if (menuDatasetExport) {
      menuDatasetExport.classList.toggle("disabled", !datasetLoaded);
    }

    if (datasetLoaded) {
      showMainUI();
      const mtInfo = mediaTypesMap[status.media_type];
      datasetInfo.textContent = mtInfo
        ? `${mtInfo.icon} ${status.num_medias} ${mtInfo.name.toLowerCase()} loaded`
        : `${status.num_medias} medias loaded`;
    } else {
      showWelcomeScreen();
    }

    return status;
  }

  function showWelcomeScreen() {
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
    const autodetectToggle = document.getElementById("autodetect-toggle");
    if (autodetectToggle) autodetectToggle.style.display = "";
    sortBar.style.display = "none";
    datasetBar.style.display = "none";
    mediaList.innerHTML = "";
    leftPanel.style.display = "none";
    if (rightPanel) rightPanel.style.display = "none";
    stripeContainer.innerHTML = "";
    if (menuFavoritesAutodetect) menuFavoritesAutodetect.classList.add("disabled");
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
    if (menuFavoritesAutodetect) menuFavoritesAutodetect.classList.remove("disabled");
  }

  function showProgress() {
    datasetOptions.style.display = "none";
    demoDatasetsDiv.style.display = "none";
    extendedImporterForm.style.display = "none";
    datasetProgress.style.display = "block";
    backButton.style.display = "none";
    const autodetectToggle = document.getElementById("autodetect-toggle");
    if (autodetectToggle) autodetectToggle.style.display = "none";
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
      showWelcomeScreen();
      vtAlert(progress.error, "warning");
      return;
    }

    if (progress.status === "idle") {
      stopProgressPolling();
      await checkDatasetStatus();
      if (datasetLoaded) {
        await fetchMedias();
        await fetchVotes();

        // Auto-select first media if none selected
        if (medias.length > 0 && !selected) {
          selectMedia(medias[0].id);
        }

        // Check if auto-detect mode is enabled
        const autodetectCheckbox = document.getElementById("autodetect-mode-checkbox");
        if (autodetectCheckbox && autodetectCheckbox.checked) {
          // Wait a moment for UI to settle
          setTimeout(async () => {
            await runAutoDetectAfterLoad();
          }, 500);
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
        btn.innerHTML = `<h3>${importer.icon || "🔌"} ${importer.display_name}</h3><p>${importer.description}</p>`;
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
          return '<span class="download-badge">Download</span>';
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
              <td class="col-name">${ds.label}</td>
              <td class="col-num">${ds.num_files}</td>
              <td class="col-num">${ds.num_categories}</td>
              <td class="col-desc" title="${ds.description.replace(/"/g, '&quot;')}">${descShort}</td>
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
        demoDatasetsDiv.innerHTML = `<div style="color:var(--color-bad); text-align:center;">Error loading demo datasets: ${e.message}</div>`;
      }
    });
    datasetLoadColumn.appendChild(demoBtnEl);

    // Always render the autodetect toggle last, after all import options
    const autodetectDiv = document.createElement("div");
    autodetectDiv.id = "autodetect-toggle";
    autodetectDiv.style = "width: 100%; margin-top: 16px; padding: 12px; background: var(--border); border-radius: 4px; box-sizing: border-box;";
    autodetectDiv.innerHTML = `
      <label style="display: flex; align-items: center; color: var(--text-primary); cursor: pointer;">
        <input type="checkbox" id="autodetect-mode-checkbox" style="margin-right: 8px;">
        <span>Run auto-detect after loading (skip manual labeling)</span>
      </label>
      <p style="font-size: 0.85rem; color: var(--text-secondary); margin: 8px 0 0 0;">When checked, automatically runs all favorite detectors and shows positive hits.</p>
    `;
    datasetWelcome.insertBefore(autodetectDiv, backButton);
  }

  function showExtendedImporterForm(importer) {
    datasetOptions.style.display = "none";
    backButton.style.display = "block";

    const inputStyle = "width:100%;padding:8px;background:var(--bg-hover);border:1px solid var(--border);border-radius:4px;color:var(--text-primary);box-sizing:border-box;";
    let html = `<div style="max-width:420px;width:100%;margin:0 auto;">`;
    html += `<h3 style="margin-bottom:16px;color:var(--text-primary);">${escapeHtml(importer.display_name)}</h3>`;
    html += `<form id="ext-imp-form">`;
    for (const field of importer.fields) {
      html += `<div style="margin-bottom:14px;">`;
      html += `<label style="display:block;margin-bottom:5px;color:var(--text-secondary);font-size:0.85rem;">${escapeHtml(field.label)}${field.required ? " *" : ""}</label>`;
      if (field.field_type === "file") {
        html += `<input type="file" name="${escapeHtml(field.key)}" accept="${escapeHtml(field.accept)}" style="color:var(--text-primary);width:100%;" ${field.required ? "required" : ""}>`;
      } else if (field.field_type === "select") {
        html += `<select name="${escapeHtml(field.key)}" style="${inputStyle}">`;
        for (const opt of field.options) {
          html += `<option value="${escapeHtml(opt)}"${opt === field.default ? " selected" : ""}>${escapeHtml(opt)}</option>`;
        }
        html += `</select>`;
      } else if (field.field_type === "folder") {
        html += `<div style="display:flex;gap:8px;align-items:center;">`;
        html += `<input type="text" name="${escapeHtml(field.key)}" placeholder="${escapeHtml(field.description)}" style="${inputStyle}flex:1;" data-folder-input="true" ${field.required ? "required" : ""}>`;
        html += `<button type="button" data-browse-btn="true" style="padding:8px 14px;background:var(--bg-hover);border:1px solid var(--border);border-radius:4px;color:var(--text-secondary);cursor:pointer;white-space:nowrap;">Browse…</button>`;
        html += `</div>`;
        html += `<input type="file" data-folder-picker="true" webkitdirectory style="display:none;">`;
      } else {
        const itype = field.field_type === "url" ? "url" : "text";
        html += `<input type="${itype}" name="${escapeHtml(field.key)}" value="${escapeHtml(field.default)}" placeholder="${escapeHtml(field.description)}" style="${inputStyle}" ${field.required ? "required" : ""}>`;
      }
      if (field.description) {
        html += `<div style="margin-top:4px;font-size:0.75rem;color:var(--text-dim);">${escapeHtml(field.description)}</div>`;
      }
      html += `</div>`;
    }
    html += `<button type="submit" style="width:100%;padding:10px;background:var(--accent);border:none;border-radius:4px;color:#fff;cursor:pointer;font-size:0.9rem;">Import</button>`;
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
    datasetOptions.style.display = "none";
    backButton.style.display = "block";

    const inputStyle = "width:100%;padding:8px;background:var(--bg-hover);border:1px solid var(--border);border-radius:4px;color:var(--text-primary);box-sizing:border-box;";

    let html = `<div style="max-width:420px;width:100%;margin:0 auto;">`;
    html += `<h3 style="margin-bottom:16px;color:var(--text-primary);">\uD83D\uDD00 Combine Existing Datasets</h3>`;
    html += `<p style="font-size:0.85rem;color:var(--text-secondary);margin-bottom:12px;">Select two or more datasets to merge. All must be the same media type. Duplicates are skipped automatically.</p>`;
    html += `<div id="combine-dataset-list" style="margin-bottom:14px;"><p style="color:var(--text-dim);font-size:0.85rem;">Loading available datasets...</p></div>`;
    html += `<div style="margin-bottom:14px;">`;
    html += `<label style="display:block;margin-bottom:5px;color:var(--text-secondary);font-size:0.85rem;">Or add a .pkl file path</label>`;
    html += `<div style="display:flex;gap:8px;">`;
    html += `<input type="text" id="combine-extra-path" placeholder="/path/to/dataset.pkl" style="${inputStyle}flex:1;">`;
    html += `<button type="button" id="combine-add-path-btn" style="padding:8px 14px;background:var(--bg-hover);border:1px solid var(--border);border-radius:4px;color:var(--text-secondary);cursor:pointer;white-space:nowrap;">Add</button>`;
    html += `</div></div>`;
    html += `<div id="combine-extra-paths" style="margin-bottom:14px;"></div>`;
    html += `<button type="button" id="combine-submit-btn" style="width:100%;padding:10px;background:var(--accent);border:none;border-radius:4px;color:#fff;cursor:pointer;font-size:0.9rem;" disabled>Select at least 2 datasets</button>`;
    html += `</div>`;

    extendedImporterForm.innerHTML = html;
    extendedImporterForm.style.display = "block";

    const listDiv = document.getElementById("combine-dataset-list");
    const submitBtn = document.getElementById("combine-submit-btn");
    const extraPathInput = document.getElementById("combine-extra-path");
    const addPathBtn = document.getElementById("combine-add-path-btn");
    const extraPathsDiv = document.getElementById("combine-extra-paths");
    const extraPaths = [];

    function updateSubmitState() {
      const checked = listDiv.querySelectorAll("input[type=checkbox]:checked");
      const total = checked.length + extraPaths.length;
      if (total >= 2) {
        submitBtn.disabled = false;
        submitBtn.textContent = `Combine ${total} Datasets`;
      } else {
        submitBtn.disabled = true;
        submitBtn.textContent = "Select at least 2 datasets";
      }
    }

    // Fetch available datasets
    try {
      const res = await fetch("/api/dataset/available-files");
      if (!res.ok) throw new Error("Failed to fetch");
      const data = await res.json();

      if (data.files.length === 0) {
        listDiv.innerHTML = `<p style="color:var(--text-dim);font-size:0.85rem;">No saved datasets found. Add file paths below.</p>`;
      } else {
        let listHtml = "";
        for (const file of data.files) {
          listHtml += `<label style="display:flex;align-items:center;padding:8px;margin-bottom:4px;background:var(--bg-hover);border-radius:4px;cursor:pointer;">`;
          listHtml += `<input type="checkbox" data-path="${escapeHtml(file.path)}" style="margin-right:10px;">`;
          listHtml += `<span style="flex:1;color:var(--text-primary);font-size:0.9rem;">${escapeHtml(file.name)}</span>`;
          listHtml += `<span style="color:var(--text-dim);font-size:0.75rem;">${file.size_mb} MB</span>`;
          listHtml += `</label>`;
        }
        listDiv.innerHTML = listHtml;
        listDiv.querySelectorAll("input[type=checkbox]").forEach(cb => {
          cb.addEventListener("change", updateSubmitState);
        });
      }
    } catch (_) {
      listDiv.innerHTML = `<p style="color:var(--text-dim);font-size:0.85rem;">Could not load available datasets. Add file paths below.</p>`;
    }

    // Handle adding extra file paths
    function addExtraPath() {
      const val = extraPathInput.value.trim();
      if (!val) return;
      extraPaths.push(val);
      extraPathInput.value = "";
      renderExtraPaths();
      updateSubmitState();
    }

    function renderExtraPaths() {
      let html = "";
      extraPaths.forEach((p, i) => {
        html += `<div style="display:flex;align-items:center;padding:6px 8px;margin-bottom:4px;background:var(--bg-hover);border-radius:4px;">`;
        html += `<span style="flex:1;color:var(--text-primary);font-size:0.85rem;word-break:break-all;">${escapeHtml(p)}</span>`;
        html += `<button type="button" data-remove-idx="${i}" style="background:none;border:none;color:var(--color-bad);cursor:pointer;font-size:1rem;padding:0 4px;">&times;</button>`;
        html += `</div>`;
      });
      extraPathsDiv.innerHTML = html;
      extraPathsDiv.querySelectorAll("[data-remove-idx]").forEach(btn => {
        btn.addEventListener("click", () => {
          extraPaths.splice(parseInt(btn.dataset.removeIdx), 1);
          renderExtraPaths();
          updateSubmitState();
        });
      });
    }

    addPathBtn.addEventListener("click", addExtraPath);
    extraPathInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") { e.preventDefault(); addExtraPath(); }
    });

    // Handle submit
    submitBtn.addEventListener("click", async () => {
      const checkedPaths = Array.from(listDiv.querySelectorAll("input[type=checkbox]:checked"))
        .map(cb => cb.dataset.path);
      const allPaths = [...checkedPaths, ...extraPaths];

      if (allPaths.length < 2) return;

      startProgressPolling();
      try {
        const res = await fetch("/api/dataset/combine", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ datasets: allPaths }),
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
    });
  }

  loadExtendedImporters();

  backButton.addEventListener("click", () => {
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

  // Dataset export
  if (menuDatasetExport && burgerDropdown) {
    menuDatasetExport.addEventListener("click", () => {
      if (menuDatasetExport.classList.contains("disabled")) return;
      window.location.href = "/api/dataset/export";
      closeBurgerMenu();
    });
  }

  // Dataset change
  if (menuDatasetChange && burgerDropdown) {
    menuDatasetChange.addEventListener("click", async () => {
      if (await vtConfirm("Changing the dataset will erase your current dataset. Continue?")) {
        fetch("/api/dataset/clear", { method: "POST" })
          .then(() => {
            medias = [];
            votes = { good: [], bad: [], click_times: {}, learned_scores: {} };
            selected = null;
            datasetLoaded = false;
            if (menuDatasetExport) menuDatasetExport.classList.add("disabled");
            updateMediaHeading();
            showWelcomeScreen();
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
        <div class="label-exporter-option" data-name="${escapeHtml(exp.name)}" style="
          background:var(--border); border:1px solid var(--bg-secondary-btn); border-radius:6px;
          padding:12px 16px; margin-bottom:10px; cursor:pointer;
          display:flex; align-items:center; gap:12px;">
          <span style="font-size:1.5rem;">${escapeHtml(exp.icon || '\uD83D\uDCE4')}</span>
          <div>
            <div style="font-weight:bold; color:var(--text-primary);">${escapeHtml(exp.display_name)}</div>
            <div style="font-size:0.8rem; color:var(--text-muted); margin-top:2px;">${escapeHtml(exp.description)}</div>
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
      closeBurgerMenu();
      await openLabelImporterModal();
    });
  }

  // Detector import
  if (menuDetectorImport && loadDetectorFile && burgerDropdown) {
    menuDetectorImport.addEventListener("click", () => {
      loadDetectorFile.click();
      closeBurgerMenu();
    });
  }

  // Detector export – open modal
  if (menuDetectorExport) {
    menuDetectorExport.addEventListener("click", () => {
      if (menuDetectorExport.classList.contains("disabled")) return;
      closeBurgerMenu();
      openDetectorExportModal();
    });
  }

  function openDetectorExportModal() {
    const optionStyle = "background:var(--border); border:1px solid var(--bg-secondary-btn); border-radius:6px; padding:12px 16px; margin-bottom:10px; cursor:pointer; display:flex; align-items:center; gap:12px;";
    detectorExportList.innerHTML = `
      <div id="detector-export-browser-btn" role="button" tabindex="0" style="${optionStyle}">
        <span style="font-size:1.5rem;">\u2B07\uFE0F</span>
        <div>
          <div style="font-weight:bold; color:var(--text-primary);">Download (Browser)</div>
          <div style="font-size:0.8rem; color:var(--text-muted); margin-top:2px;">Download the detector file directly to your browser.</div>
        </div>
      </div>
      <div id="detector-export-server-btn" role="button" tabindex="0" style="${optionStyle}">
        <span style="font-size:1.5rem;">\uD83D\uDCBE</span>
        <div>
          <div style="font-weight:bold; color:var(--text-primary);">Save to Server</div>
          <div style="font-size:0.8rem; color:var(--text-muted); margin-top:2px;">Save the detector file to the server disk.</div>
        </div>
      </div>
    `;

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

    detectorExportModal.classList.add("show");
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
      favoritesList.innerHTML = '<p style="color: #888;">No favorite detectors saved yet.</p>';
      return;
    }

    const mediaIcons = Object.fromEntries(Object.entries(mediaTypesMap).map(([k, v]) => [k, v.icon]));
    favoritesList.innerHTML = favoriteDetectors.map(detector => {
      const icon = mediaIcons[detector.media_type] || "🔍";
      const created = detector.created_at
        ? new Date(detector.created_at * 1000).toLocaleDateString()
        : "";
      return `
      <div style="background: var(--border); padding: 12px; margin-bottom: 8px; border-radius: 4px; display: flex; justify-content: space-between; align-items: center; gap: 12px;">
        <div style="flex: 1; min-width: 0;">
          <div style="font-weight: bold; color: var(--accent); white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">${escapeHtml(detector.name)}</div>
          <div style="font-size: 0.8rem; color: var(--text-muted); margin-top: 3px; display: flex; align-items: center; gap: 10px;">
            <span style="background: var(--bg-surface); border: 1px solid var(--bg-secondary-btn); border-radius: 3px; padding: 1px 6px; white-space: nowrap;">${icon} ${detector.media_type}</span>
            <span>threshold&nbsp;${detector.threshold.toFixed(2)}</span>
            ${created ? `<span>${created}</span>` : ""}
          </div>
        </div>
        <div style="display: flex; gap: 6px; flex-shrink: 0;">
          <button onclick="renameDetector('${escapeHtml(detector.name)}')" style="padding: 4px 10px; background: var(--bg-secondary-btn); color: var(--text-btn-secondary); border: 1px solid var(--border-secondary); border-radius: 4px; cursor: pointer; font-size: 0.78rem;">Rename</button>
          <button onclick="deleteDetector('${escapeHtml(detector.name)}')" style="padding: 4px 10px; background: #c0392b; color: #fff; border: none; border-radius: 4px; cursor: pointer; font-size: 0.78rem;">Delete</button>
        </div>
      </div>`;
    }).join('');
  }

  function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML.replace(/'/g, '&#39;');
  }

  window.renameDetector = async function(oldName) {
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
  };

  window.deleteDetector = async function(name) {
    if (!await vtConfirm(`Are you sure you want to delete detector "${name}"?`)) return;

    const res = await fetch(`/api/favorite-detectors/${encodeURIComponent(name)}`, {
      method: "DELETE",
    });

    if (res.ok) {
      await loadFavoriteDetectors();
    } else {
      await vtAlert("Failed to delete detector.", "error");
    }
  };

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
        setFavAddStatus("Need at least one good and one bad vote first.", "#f44336");
        return;
      }
      const name = favAddName ? favAddName.value.trim() : "";
      if (!name) {
        setFavAddStatus("Enter a name first.", "#f44336");
        return;
      }
      setFavAddStatus("Training detector\u2026", "#aaa");

      const exportRes = await fetch("/api/detector/export", { method: "POST" });
      if (!exportRes.ok) {
        setFavAddStatus("Failed to train detector.", "#f44336");
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
        setFavAddStatus(`Detector \u201c${name}\u201d saved (${mediaType}).`, "#4caf50");
        if (favAddName) favAddName.value = "";
        await loadFavoriteDetectors();
      } else {
        setFavAddStatus("Failed to save detector.", "#f44336");
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

    const btnStyle =
      "flex: 1; min-width: 140px; padding: 8px 12px; background: var(--bg-secondary-btn); color: var(--text-btn-secondary); border: 1px solid var(--border-secondary); border-radius: 4px; cursor: pointer; font-size: 0.8rem;";
    const headerStyle =
      "font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.04em; color: var(--text-dim); margin-top: 10px; margin-bottom: 4px;";

    // Helper: add a section header + row of buttons
    function addSection(title, buttons) {
      if (buttons.length === 0) return;
      const header = document.createElement("div");
      header.style.cssText = headerStyle;
      header.textContent = title;
      favImporterButtonsDiv.appendChild(header);
      const row = document.createElement("div");
      row.style.cssText = "display: flex; gap: 8px; flex-wrap: wrap;";
      for (const btn of buttons) row.appendChild(btn);
      favImporterButtonsDiv.appendChild(row);
    }

    // Helper: create a file-picker button for a processor importer
    function makeProcFileButton(imp, fileField) {
      const btn = document.createElement("button");
      btn.textContent = `${imp.icon || "\u{1F9E9}"} ${imp.display_name}`;
      btn.style.cssText = btnStyle;
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
          setFavAddStatus(`Importing from ${imp.display_name}\u2026`, "#aaa");
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
            setFavAddStatus(`Saved \u201c${data.name}\u201d (${data.media_type}${detail}).`, "#4caf50");
            if (favAddName) favAddName.value = "";
            await loadFavoriteDetectors();
          } else {
            const err = await res.json().catch(() => ({}));
            setFavAddStatus(`Error: ${err.error || "Import failed"}`, "#f44336");
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
      btn.style.cssText = btnStyle;
      btn.addEventListener("click", async () => {
        const value = await vtPrompt(`Enter ${textField.label}:`, textField.placeholder || "");
        if (!value) return;
        const defaultName = value.split("/").pop().replace(/\.[^/.]+$/, "");
        const detectorName = (favAddName && favAddName.value.trim()) || defaultName;
        setFavAddStatus(`Importing from ${imp.display_name}\u2026`, "#aaa");
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
          setFavAddStatus(`Saved \u201c${data.name}\u201d (${data.media_type}${detail}).`, "#4caf50");
          if (favAddName) favAddName.value = "";
          await loadFavoriteDetectors();
        } else {
          const err = await res.json().catch(() => ({}));
          setFavAddStatus(`Error: ${err.error || "Import failed"}`, "#f44336");
        }
      });
      return btn;
    }

    // Helper: create a file-picker button for a label importer (trains detector)
    function makeLabelFileButton(imp, fileField) {
      const btn = document.createElement("button");
      btn.textContent = `${imp.icon || "\u{1F3F7}\uFE0F"} ${imp.display_name}`;
      btn.style.cssText = btnStyle;
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
          setFavAddStatus(`Training from labelset (${imp.display_name})\u2026`, "#aaa");
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
            setFavAddStatus(`Trained \u201c${data.name}\u201d (${data.media_type}${detail}).`, "#4caf50");
            if (favAddName) favAddName.value = "";
            await loadFavoriteDetectors();
          } else {
            const err = await res.json().catch(() => ({}));
            setFavAddStatus(`Error: ${err.error || "Training failed"}`, "#f44336");
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
      btn.style.cssText = btnStyle;
      btn.addEventListener("click", async () => {
        const value = await vtPrompt(`Enter ${textField.label}:`, textField.placeholder || "");
        if (!value) return;
        const defaultName = value.split("/").pop().replace(/\.[^/.]+$/, "");
        const detectorName = (favAddName && favAddName.value.trim()) || defaultName;
        setFavAddStatus(`Training from labelset (${imp.display_name})\u2026`, "#aaa");
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
          setFavAddStatus(`Trained \u201c${data.name}\u201d (${data.media_type}${detail}).`, "#4caf50");
          if (favAddName) favAddName.value = "";
          await loadFavoriteDetectors();
        } else {
          const err = await res.json().catch(() => ({}));
          setFavAddStatus(`Error: ${err.error || "Training failed"}`, "#f44336");
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

  if (menuFavoritesAutodetect) {
    menuFavoritesAutodetect.addEventListener("click", async () => {
      if (menuFavoritesAutodetect.classList.contains("disabled")) return;

      closeBurgerMenu();

      // Show progress modal
      autodetectProgressModal.classList.add("show");
      autodetectProgressText.textContent = "Running auto-detect...";
      autodetectProgressBar.style.width = "0%";

      // Simulate progress (since we don't have real-time progress from backend)
      let progress = 0;
      const progressInterval = setInterval(() => {
        progress += 5;
        if (progress > 90) progress = 90;
        autodetectProgressBar.style.width = `${progress}%`;
      }, 200);

      // Run auto-detect
      const res = await fetch("/api/auto-detect", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      });

      clearInterval(progressInterval);
      autodetectProgressBar.style.width = "100%";

      setTimeout(async () => {
        autodetectProgressModal.classList.remove("show");

        if (!res.ok) {
          await vtAlert("Auto-detect failed. Make sure you have saved some favorite detectors for this media type.", "error");
          return;
        }

        res.json().then(data => {
          displayAutodetectResults(data);
        });
      }, 500);
    });
  }

  async function runAutoDetectAfterLoad() {
    if (medias.length === 0) {
      return;
    }

    // Show progress modal
    autodetectProgressModal.classList.add("show");
    autodetectProgressText.textContent = "Running auto-detect...";
    autodetectProgressBar.style.width = "0%";

    // Simulate progress (since we don't have real-time progress from backend)
    let progress = 0;
    const progressInterval = setInterval(() => {
      progress += 5;
      if (progress > 90) progress = 90;
      autodetectProgressBar.style.width = `${progress}%`;
    }, 200);

    // Run auto-detect
    const res = await fetch("/api/auto-detect", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
    });

    clearInterval(progressInterval);
    autodetectProgressBar.style.width = "100%";

    setTimeout(async () => {
      autodetectProgressModal.classList.remove("show");

      if (!res.ok) {
        await vtAlert("Auto-detect failed. Make sure you have saved some favorite detectors for this media type.", "error");
        return;
      }

      res.json().then(data => {
        displayAutodetectResults(data);
      });
    }, 500);
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
      let tableHtml = `<table style="width: 100%; border-collapse: collapse; font-size: 0.85rem;">`;
      tableHtml += `<thead><tr style="border-bottom: 1px solid var(--bg-secondary-btn);">`;
      tableHtml += `<th style="text-align: left; padding: 8px; color: var(--accent);">Origin</th>`;
      tableHtml += `<th style="text-align: left; padding: 8px; color: var(--accent);">Name</th>`;
      tableHtml += `<th style="text-align: left; padding: 8px; color: var(--accent);">MD5</th>`;
      tableHtml += `<th style="text-align: left; padding: 8px; color: var(--accent);">Filename</th>`;
      tableHtml += `</tr></thead><tbody>`;
      for (const hit of allHits) {
        const origin = escapeHtml(formatOrigin(hit));
        const name = escapeHtml(hit.origin_name || hit.filename || "");
        const md5 = escapeHtml(hit.md5 || "");
        const filename = escapeHtml(hit.filename || "");
        tableHtml += `<tr style="border-bottom: 1px solid var(--border);">`;
        tableHtml += `<td style="padding: 6px 8px; color: var(--text-secondary);">${origin}</td>`;
        tableHtml += `<td style="padding: 6px 8px; color: var(--text-primary);">${name}</td>`;
        tableHtml += `<td style="padding: 6px 8px; color: var(--text-muted); font-family: monospace; font-size: 0.75rem;">${md5}</td>`;
        tableHtml += `<td style="padding: 6px 8px; color: var(--text-secondary);">${filename}</td>`;
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
    const inputStyle = "width:100%;padding:6px 8px;background:var(--bg-hover);border:1px solid var(--border);border-radius:4px;color:var(--text-primary);box-sizing:border-box;font-size:0.85rem;";
    let html = "";
    for (const field of exp.fields) {
      html += `<div style="margin-bottom:8px;">`;
      html += `<label style="display:block;margin-bottom:3px;color:var(--text-secondary);font-size:0.8rem;">${escapeHtml(field.label)}${field.required ? " *" : ""}</label>`;
      if (field.field_type === "select") {
        html += `<select name="${escapeHtml(field.key)}" data-export-field style="${inputStyle}">`;
        for (const opt of field.options) {
          html += `<option value="${escapeHtml(opt)}"${opt === field.default ? " selected" : ""}>${escapeHtml(opt)}</option>`;
        }
        html += `</select>`;
      } else {
        const itype = field.field_type === "password" ? "password" : (field.field_type === "email" ? "email" : "text");
        const placeholder = escapeHtml(field.placeholder || field.description || "");
        html += `<input type="${itype}" name="${escapeHtml(field.key)}" value="${escapeHtml(field.default)}" placeholder="${placeholder}" data-export-field style="${inputStyle}" ${field.required ? "required" : ""}>`;
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
          setExportStatus("Failed to compute fill counts.", "#f44336");
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
          setExportStatus("Failed to fill labels.", "#f44336");
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
        setExportStatus(data.message || "Export complete.", "#4caf50");
      } else {
        setExportStatus(data.error || "Export failed.", "#f44336");
      }
    } catch (err) {
      setExportStatus(`Export error: ${err.message}`, "#f44336");
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
        const importableKeys = ["volume", "theme", "inclusion", "enrich_descriptions", "safe_thresholds", "calibrate_count", "calibration_fraction", "swipe_animation", "show_thumbnails_left", "show_thumbnails_right"];
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
        // Immediately prompt to select a detector file
        sortMode = radio.value;
        textSortWrap.style.display = "none";
        loadSortWrap.style.display = "";
        sortStatus.textContent = "";
        // Trigger file picker
        loadDetectorFile.click();
        return;
      }

      sortMode = radio.value;
      textSortWrap.style.display = sortMode === "text" ? "" : "none";
      loadSortWrap.style.display = sortMode === "load" ? "" : "none";
      sortStatus.textContent = "";

      if (sortMode === "text") {
        onTextSortInput();
      } else if (sortMode === "learned") {
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
      sortStatus.textContent = "Load a detector first";
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

  loadDetectorBtn.addEventListener("click", () => {
    loadDetectorFile.click();
  });

  loadDetectorFile.addEventListener("change", async () => {
    const file = loadDetectorFile.files[0];
    if (!file) {
      // User cancelled - revert to text mode if no detector loaded
      if (loadedDetector === null) {
        document.querySelector('input[name="sort-mode"][value="text"]').checked = true;
        sortMode = "text";
        textSortWrap.style.display = "";
        loadSortWrap.style.display = "none";
        sortStatus.textContent = "";
      }
      return;
    }
    sortStatus.textContent = "Loading detector\u2026";
    menuDetectorStatus.textContent = "Loading detector\u2026";
    const text = await file.text();
    try {
      loadedDetector = JSON.parse(text);
      sortStatus.textContent = "Detector loaded";
      menuDetectorStatus.textContent = "Detector loaded";
      setTimeout(() => { menuDetectorStatus.textContent = ""; }, 3000);
      updateSortModeAvailability();
      // Ensure load mode is selected
      document.querySelector('input[name="sort-mode"][value="load"]').checked = true;
      sortMode = "load";
      loadSortWrap.style.display = "";
      textSortWrap.style.display = "none";
      fetchLoadedSort(true);
    } catch (e) {
      sortStatus.textContent = "Invalid detector file";
      menuDetectorStatus.textContent = "Invalid detector file";
      setTimeout(() => { menuDetectorStatus.textContent = ""; }, 3000);
      loadedDetector = null;
      updateSortModeAvailability();
      // Revert to text mode on error
      document.querySelector('input[name="sort-mode"][value="text"]').checked = true;
      sortMode = "text";
      textSortWrap.style.display = "";
      loadSortWrap.style.display = "none";
    }
    loadDetectorFile.value = "";
  });

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

  // Map from type_id to plural display name (matches MediaType.folder_import_name)
  const MEDIA_TYPE_NAMES = {
    audio: "Sounds",
    image: "Images",
    video: "Videos",
    paragraph: "Paragraphs",
  };

  function updateMediaHeading() {
    const heading = document.getElementById("media-heading");
    if (!heading) return;
    if (!medias || medias.length === 0) {
      heading.textContent = "Medias";
      return;
    }
    const types = new Set(medias.map(m => m.type));
    if (types.size === 1) {
      const type = types.values().next().value;
      heading.textContent = MEDIA_TYPE_NAMES[type] || "Medias";
    } else {
      heading.textContent = "Medias";
    }
  }

  async function fetchMedias() {
    const res = await fetch("/api/medias");
    medias = await res.json();
    updateMediaHeading();
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
          html += `<img class="media-thumbnail" src="${thumbnailUrl(c)}" alt="${mediaLabel}" loading="lazy">`;
        }
        html += `<div class="media-thumb-info">`;
      }
      html += `<div style="font-weight: 500;">${mediaLabel}</div>`;
      if (scoreMap[c.id] !== undefined) {
        html += `<span class="sim">${(scoreMap[c.id] * 100).toFixed(1)}%</span>`;
      }
      let subInfo = [];
      if (c.frequency) {
        subInfo.push(`${c.frequency} Hz`);
      }
      if (c.category && c.category !== "unknown") {
        subInfo.push(c.category);
      }
      if (c.duration && c.duration > 0) {
        subInfo.push(`${c.duration.toFixed(1)}s`);
      }
      if (c.width && c.height) {
        subInfo.push(`${c.width}×${c.height}`);
      }
      if (c.word_count) {
        subInfo.push(`${c.word_count} words`);
      }
      html += `<div class="sub">${subInfo.join(' &middot; ')}</div>`;
      if (useThumbnail) {
        html += `</div>`;
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
    let metaInfo = [];
    if (c.frequency) {
      metaInfo.push(`${c.frequency} Hz`);
    }
    if (c.category && c.category !== "unknown") {
      metaInfo.push(c.category);
    }
    // Use media type metadata from the registry for duration/details display
    const mtInfo = mediaTypesMap[mediaType];
    if (c.duration && c.duration > 0) {
      metaInfo.push(`${c.duration.toFixed(1)}s`);
    }
    if (c.width && c.height) {
      metaInfo.push(`${c.width}×${c.height}`);
    }
    if (c.word_count) {
      metaInfo.push(`${c.word_count} words`);
    }
    metaInfo.push(`${(c.file_size / 1024).toFixed(1)} KB`);

    // Render media player based on media type.
    // Known types get specialised players; new/unknown types fall back to
    // the generic /api/medias/{id}/media endpoint.
    let playerHTML = '';
    if (mediaType === "video") {
      playerHTML = `<video controls loop autoplay src="/api/medias/${c.id}/video" id="media-video" aria-label="${escapeHtml(c.filename || 'Video media')}" style="width: 600px; max-height: 400px; border: 1px solid var(--border); border-radius: 8px; background: var(--bg-surface);"></video>`;
    } else if (mediaType === "image") {
      playerHTML = `<div style="flex: 1; min-height: 0; width: 100%; display: flex; align-items: center; justify-content: center;"><img src="/api/medias/${c.id}/image" id="media-image" alt="${escapeHtml(c.filename || 'Image media')}" style="max-width: 100%; max-height: 100%; object-fit: contain; border: 1px solid var(--border); border-radius: 8px; background: var(--bg-surface);"></div>`;
    } else if (mediaType === "paragraph") {
      playerHTML = `
        <div id="media-paragraph" style="max-width: 600px; max-height: 400px; overflow-y: auto; padding: 16px; border: 1px solid var(--border); border-radius: 8px; background: var(--bg-surface); white-space: pre-wrap; line-height: 1.6; text-align: left;">
          Loading...
        </div>`;
    } else if (mediaType === "audio") {
      // Audio/Sound
      playerHTML = `
        <canvas id="waveform-canvas" width="600" height="120" role="img" aria-label="Audio waveform visualization"></canvas>
        <audio controls controlslist="nodownload" loop autoplay src="/api/medias/${c.id}/audio" id="media-audio" aria-label="${escapeHtml(c.filename || 'Audio media')}"></audio>`;
    } else {
      // Unknown/new media type: try to render via generic endpoint.
      // If it loops, use a video element; otherwise use a generic embed.
      const loops = mtInfo && mtInfo.loops;
      if (loops) {
        playerHTML = `<video controls loop autoplay src="/api/medias/${c.id}/media" id="media-video" style="width: 600px; max-height: 400px; border: 1px solid var(--border); border-radius: 8px; background: var(--bg-surface);"></video>`;
      } else {
        playerHTML = `<div style="flex: 1; min-height: 0; width: 100%; display: flex; align-items: center; justify-content: center;"><object data="/api/medias/${c.id}/media" style="max-width: 100%; max-height: 100%; border: 1px solid var(--border); border-radius: 8px;">${escapeHtml(c.filename || 'Media')}</object></div>`;
      }
    }

    center.innerHTML = `
      <div class="meta">
        <h2>${c.filename || 'Media #' + c.id}</h2>
        <p>${metaInfo.join(' &middot; ')}</p>
      </div>
      <div class="media-swipe-wrapper" id="media-swipe-wrapper">
        ${playerHTML}
      </div>
      <div class="metadata-grid">
        ${c.frequency ? `
        <div class="metadata-item">
          <span class="metadata-label">Frequency</span>
          <span class="metadata-value">${c.frequency} Hz</span>
        </div>` : ''}
        ${c.category && c.category !== 'unknown' ? `
        <div class="metadata-item">
          <span class="metadata-label">Category</span>
          <span class="metadata-value">${c.category}</span>
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
          <span class="metadata-label">Filename</span>
          <span class="metadata-value">${c.filename || 'media_' + c.id + '.wav'}</span>
        </div>
        <div class="metadata-item">
          <span class="metadata-label">MD5</span>
          <span class="metadata-value metadata-md5">${c.md5}</span>
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
  }

  async function castVote(id, vote) {
    if (isVoting) return; // Prevent double-click from toggling the vote off
    isVoting = true;
    try {
      const mediaName = (medias.find(c => c.id === id) || {}).filename || `Clip #${id}`;
      await fetch(`/api/medias/${id}/vote`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ vote }),
      });
      announce(`Voted ${vote} on ${mediaName}`);
      await fetchVotes();

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
        <div class="label-importer-option" data-name="${escapeHtml(imp.name)}" style="
          background:var(--border); border:1px solid var(--bg-secondary-btn); border-radius:6px;
          padding:12px 16px; margin-bottom:10px; cursor:pointer;
          display:flex; align-items:center; gap:12px;">
          <span style="font-size:1.5rem;">${escapeHtml(imp.icon || '🏷️')}</span>
          <div>
            <div style="font-weight:bold; color:var(--text-primary);">${escapeHtml(imp.display_name)}</div>
            <div style="font-size:0.8rem; color:var(--text-muted); margin-top:2px;">${escapeHtml(imp.description)}</div>
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
    promptDiv.style.cssText = "margin-top:14px;padding:14px;background:var(--border);border:1px solid var(--bg-secondary-btn);border-radius:6px;";
    promptDiv.innerHTML = `
      <div style="color:var(--text-primary);margin-bottom:10px;font-size:0.9rem;">
        <strong>${n}</strong> element(s) from the labelset were not found in your dataset.
        Import them from their origins?
      </div>
      <div style="display:flex;gap:10px;">
        <button id="missing-import-btn" style="flex:1;padding:8px;background:var(--accent);border:none;border-radius:4px;color:#fff;cursor:pointer;font-size:0.85rem;">Import medias</button>
        <button id="missing-skip-btn" style="flex:1;padding:8px;background:var(--bg-secondary-btn);border:none;border-radius:4px;color:var(--text-btn-secondary);cursor:pointer;font-size:0.85rem;">Skip</button>
      </div>
      <div id="missing-status" style="min-height:1.4em;font-size:0.85rem;color:var(--text-muted);margin-top:8px;"></div>
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

    const inputStyle = "width:100%;padding:8px;background:var(--bg-hover);border:1px solid var(--border);border-radius:4px;color:var(--text-primary);box-sizing:border-box;";
    let html = `<h3 style="margin-bottom:14px;color:var(--text-primary);">${escapeHtml(importer.display_name)}</h3>`;
    html += `<form id="label-imp-form">`;
    for (const field of importer.fields) {
      html += `<div style="margin-bottom:14px;">`;
      html += `<label style="display:block;margin-bottom:5px;color:var(--text-secondary);font-size:0.85rem;">${escapeHtml(field.label)}${field.required ? " *" : ""}</label>`;
      if (field.field_type === "file") {
        html += `<input type="file" name="${escapeHtml(field.key)}" accept="${escapeHtml(field.accept)}" style="color:var(--text-primary);width:100%;" ${field.required ? "required" : ""}>`;
      } else if (field.field_type === "select") {
        html += `<select name="${escapeHtml(field.key)}" style="${inputStyle}">`;
        for (const opt of field.options) {
          html += `<option value="${escapeHtml(opt)}"${opt === field.default ? " selected" : ""}>${escapeHtml(opt)}</option>`;
        }
        html += `</select>`;
      } else {
        const itype = field.field_type === "password" ? "password" : "text";
        const placeholder = escapeHtml(field.placeholder || field.description);
        html += `<input type="${itype}" name="${escapeHtml(field.key)}" value="${escapeHtml(field.default)}" placeholder="${placeholder}" style="${inputStyle}" ${field.required ? "required" : ""}>`;
      }
      if (field.description) {
        html += `<div style="margin-top:4px;font-size:0.75rem;color:var(--text-dim);">${escapeHtml(field.description)}</div>`;
      }
      html += `</div>`;
    }
    html += `<div id="label-imp-status" style="min-height:1.4em;font-size:0.85rem;color:var(--text-muted);margin-bottom:10px;"></div>`;
    html += `<button type="submit" style="width:100%;padding:10px;background:var(--accent);border:none;border-radius:4px;color:#fff;cursor:pointer;font-size:0.9rem;">Import</button>`;
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
        <div class="processor-importer-option" data-name="${escapeHtml(imp.name)}" style="
          background:var(--border); border:1px solid var(--bg-secondary-btn); border-radius:6px;
          padding:12px 16px; margin-bottom:10px; cursor:pointer;
          display:flex; align-items:center; gap:12px;">
          <span style="font-size:1.5rem;">${escapeHtml(imp.icon || '\u{1F9E9}')}</span>
          <div>
            <div style="font-weight:bold; color:var(--text-primary);">${escapeHtml(imp.display_name)}</div>
            <div style="font-size:0.8rem; color:var(--text-muted); margin-top:2px;">${escapeHtml(imp.description)}</div>
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

    const inputStyle = "width:100%;padding:8px;background:var(--bg-hover);border:1px solid var(--border);border-radius:4px;color:var(--text-primary);box-sizing:border-box;";
    let html = `<h3 style="margin-bottom:14px;color:var(--text-primary);">${escapeHtml(importer.display_name)}</h3>`;
    html += `<form id="proc-imp-form">`;
    // Name field (always required)
    html += `<div style="margin-bottom:14px;">`;
    html += `<label style="display:block;margin-bottom:5px;color:var(--text-secondary);font-size:0.85rem;">Detector Name *</label>`;
    html += `<input type="text" name="name" placeholder="e.g. Dog Barks" style="${inputStyle}" required>`;
    html += `<div style="margin-top:4px;font-size:0.75rem;color:var(--text-dim);">Name for the imported detector.</div>`;
    html += `</div>`;
    for (const field of importer.fields) {
      html += `<div style="margin-bottom:14px;">`;
      html += `<label style="display:block;margin-bottom:5px;color:var(--text-secondary);font-size:0.85rem;">${escapeHtml(field.label)}${field.required ? " *" : ""}</label>`;
      if (field.field_type === "file") {
        html += `<input type="file" name="${escapeHtml(field.key)}" accept="${escapeHtml(field.accept)}" style="color:var(--text-primary);width:100%;" ${field.required ? "required" : ""}>`;
      } else if (field.field_type === "select") {
        html += `<select name="${escapeHtml(field.key)}" style="${inputStyle}">`;
        for (const opt of field.options) {
          html += `<option value="${escapeHtml(opt)}"${opt === field.default ? " selected" : ""}>${escapeHtml(opt || "(auto-detect)")}</option>`;
        }
        html += `</select>`;
      } else {
        const itype = field.field_type === "password" ? "password" : "text";
        const placeholder = escapeHtml(field.placeholder || field.description);
        html += `<input type="${itype}" name="${escapeHtml(field.key)}" value="${escapeHtml(field.default)}" placeholder="${placeholder}" style="${inputStyle}" ${field.required ? "required" : ""}>`;
      }
      if (field.description) {
        html += `<div style="margin-top:4px;font-size:0.75rem;color:var(--text-dim);">${escapeHtml(field.description)}</div>`;
      }
      html += `</div>`;
    }
    html += `<div id="proc-imp-status" style="min-height:1.4em;font-size:0.85rem;color:var(--text-muted);margin-bottom:10px;"></div>`;
    html += `<button type="submit" style="width:100%;padding:10px;background:var(--accent);border:none;border-radius:4px;color:#fff;cursor:pointer;font-size:0.9rem;">Import</button>`;
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

  // Update label counts and schedule an indicator refresh
  function updateLabelCounts() {
    goodCountSpan.textContent = `(${votes.good.length})`;
    badCountSpan.textContent = `(${votes.bad.length})`;
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
    }

    document.getElementById("smart-section").style.display = "none";
    document.getElementById("stable-section").style.display = "none";
    document.getElementById("span-section").style.display = "";
    document.getElementById("progress-modal-title").textContent = "Span: Diversity Coverage";

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
      document.getElementById("progress-modal-title").textContent = "Span: Diversity Coverage";
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

    // Extract data (skip first entry since it has no previous to compare)
    const dataToPlot = stabilityData.slice(1);
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
      announce(`${theme === "light" ? "Light" : "Dark"} mode enabled`);
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
