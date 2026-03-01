/**
 * Settings module — modal controls, theme switching, persistence.
 */

import State from "./state.js";
import { applyTheme, vtAlert } from "./ui.js";
import { putJSON } from "./api.js";

// ---- DOM refs ----------------------------------------------------------

const settingsModal             = document.getElementById("settings-modal");
const settingsModalClose        = document.getElementById("settings-modal-close");
const menuSettings              = document.getElementById("menu-settings");
const safeThresholdsCheckbox    = document.getElementById("safe-thresholds-checkbox");
const enrichDescCheckbox        = document.getElementById("enrich-descriptions-checkbox");
const settingsDefaultBtn        = document.getElementById("settings-default-btn");
const settingsImportBtn         = document.getElementById("settings-import-btn");
const settingsImportFile        = document.getElementById("settings-import-file");
const settingsExportBtn         = document.getElementById("settings-export-btn");
const swipeAnimationCheckbox    = document.getElementById("swipe-animation-checkbox");
const showThumbnailsLeftCheckbox  = document.getElementById("show-thumbnails-left-checkbox");
const showThumbnailsRightCheckbox = document.getElementById("show-thumbnails-right-checkbox");
const calibrateCountInput       = document.getElementById("calibrate-count-input");
const calibrationFractionInput  = document.getElementById("calibration-fraction-input");
const inclusionSlider           = document.getElementById("inclusion-slider");
const inclusionValue            = document.getElementById("inclusion-value");
const autopilotTopGreensInput   = document.getElementById("autopilot-top-greens-input");
const autopilotHardRedsInput    = document.getElementById("autopilot-hard-reds-input");
const favMtCheckboxes           = document.querySelectorAll("[data-media-type]");
const themeBtns                 = document.querySelectorAll(".theme-btn");

// ---- Populate settings modal -------------------------------------------

export function populateSettingsModal(data) {
  applyTheme(data.theme || "dark");
  if (calibrateCountInput) calibrateCountInput.value = data.calibrate_count;
  if (calibrationFractionInput) calibrationFractionInput.value = data.calibration_fraction;
  if (safeThresholdsCheckbox) safeThresholdsCheckbox.checked = !!data.safe_thresholds;
  if (enrichDescCheckbox) enrichDescCheckbox.checked = !!data.enrich_descriptions;
  if (swipeAnimationCheckbox) {
    const val = data.swipe_animation !== undefined ? !!data.swipe_animation : true;
    swipeAnimationCheckbox.checked = val;
    State.swipeAnimation = val;
  }
  if (showThumbnailsLeftCheckbox) {
    showThumbnailsLeftCheckbox.checked = !!data.show_thumbnails_left;
    State.showThumbnailsLeft = !!data.show_thumbnails_left;
  }
  if (showThumbnailsRightCheckbox) {
    const val = data.show_thumbnails_right !== undefined ? !!data.show_thumbnails_right : true;
    showThumbnailsRightCheckbox.checked = val;
    State.showThumbnailsRight = val;
  }
  if (autopilotTopGreensInput) autopilotTopGreensInput.value = data.autopilot_top_greens;
  if (autopilotHardRedsInput) autopilotHardRedsInput.value = data.autopilot_hard_reds;
  const favList = data.autoload_media_types || [];
  favMtCheckboxes.forEach((cb) => {
    cb.checked = favList.includes(cb.dataset.mediaType);
  });
}

// ---- Load settings at startup ------------------------------------------

export async function loadSettings() {
  try {
    const res = await fetch("/api/settings");
    if (!res.ok) return;
    const data = await res.json();
    if (typeof data.volume === "number") {
      State.audioVolume = data.volume;
      const audioEl = document.getElementById("media-audio");
      if (audioEl) audioEl.volume = State.audioVolume;
    }
    if (data.theme) applyTheme(data.theme);
    if (enrichDescCheckbox) enrichDescCheckbox.checked = !!data.enrich_descriptions;
    if (typeof data.inclusion === "number") {
      State.inclusion = data.inclusion;
      if (inclusionSlider) inclusionSlider.value = State.inclusion;
      if (inclusionValue) inclusionValue.textContent = State.inclusion;
    }
    if (calibrateCountInput && typeof data.calibrate_count === "number") calibrateCountInput.value = data.calibrate_count;
    if (calibrationFractionInput && typeof data.calibration_fraction === "number") calibrationFractionInput.value = data.calibration_fraction;
    if (safeThresholdsCheckbox) safeThresholdsCheckbox.checked = !!data.safe_thresholds;
    if (data.swipe_animation !== undefined) {
      State.swipeAnimation = !!data.swipe_animation;
      if (swipeAnimationCheckbox) swipeAnimationCheckbox.checked = State.swipeAnimation;
    }
    if (showThumbnailsLeftCheckbox) {
      showThumbnailsLeftCheckbox.checked = !!data.show_thumbnails_left;
      State.showThumbnailsLeft = !!data.show_thumbnails_left;
    }
    if (showThumbnailsRightCheckbox) {
      const val = data.show_thumbnails_right !== undefined ? !!data.show_thumbnails_right : true;
      showThumbnailsRightCheckbox.checked = val;
      State.showThumbnailsRight = val;
    }
    if (autopilotTopGreensInput && typeof data.autopilot_top_greens === "number") autopilotTopGreensInput.value = data.autopilot_top_greens;
    if (autopilotHardRedsInput && typeof data.autopilot_hard_reds === "number") autopilotHardRedsInput.value = data.autopilot_hard_reds;
  } catch (_) {
    // Settings not available yet; use defaults
  }
}

// ---- Save helpers ------------------------------------------------------

export function saveVolume(vol) {
  if (State.volumeSaveTimer) clearTimeout(State.volumeSaveTimer);
  State.volumeSaveTimer = setTimeout(() => {
    fetch("/api/settings", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ volume: vol }),
    }).catch(() => {});
  }, 500);
}

export function saveTheme(theme) {
  fetch("/api/settings", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ theme }),
  }).catch(() => {});
}

// ---- Wire all settings event listeners ---------------------------------

export function initSettingsListeners(callbacks) {
  const { renderMediaList, renderVotes, onTextSortInput } = callbacks;

  const burgerDropdown = document.getElementById("burger-dropdown");

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
    settingsModalClose.addEventListener("click", () => settingsModal.classList.remove("show"));
  }

  if (safeThresholdsCheckbox) {
    safeThresholdsCheckbox.addEventListener("change", () => {
      putJSON("/api/settings", { safe_thresholds: safeThresholdsCheckbox.checked }).catch(() => {});
    });
  }

  if (swipeAnimationCheckbox) {
    swipeAnimationCheckbox.addEventListener("change", () => {
      State.swipeAnimation = swipeAnimationCheckbox.checked;
      putJSON("/api/settings", { swipe_animation: swipeAnimationCheckbox.checked }).catch(() => {});
    });
  }

  if (enrichDescCheckbox) {
    enrichDescCheckbox.addEventListener("change", () => {
      putJSON("/api/settings", { enrich_descriptions: enrichDescCheckbox.checked }).catch(() => {});
      if (State.sortMode === "text") onTextSortInput();
    });
  }

  if (showThumbnailsLeftCheckbox) {
    showThumbnailsLeftCheckbox.addEventListener("change", () => {
      State.showThumbnailsLeft = showThumbnailsLeftCheckbox.checked;
      putJSON("/api/settings", { show_thumbnails_left: State.showThumbnailsLeft }).catch(() => {});
      renderMediaList();
    });
  }

  if (showThumbnailsRightCheckbox) {
    showThumbnailsRightCheckbox.addEventListener("change", () => {
      State.showThumbnailsRight = showThumbnailsRightCheckbox.checked;
      putJSON("/api/settings", { show_thumbnails_right: State.showThumbnailsRight }).catch(() => {});
      renderVotes();
    });
  }

  favMtCheckboxes.forEach((cb) => {
    cb.addEventListener("change", () => {
      const selected = [];
      favMtCheckboxes.forEach((c) => { if (c.checked) selected.push(c.dataset.mediaType); });
      putJSON("/api/settings", { autoload_media_types: selected }).catch(() => {});
    });
  });

  if (calibrateCountInput) {
    calibrateCountInput.addEventListener("change", () => {
      const val = Math.max(1, Math.min(100, parseInt(calibrateCountInput.value) || 2));
      calibrateCountInput.value = val;
      putJSON("/api/settings", { calibrate_count: val }).catch(() => {});
    });
  }

  if (calibrationFractionInput) {
    calibrationFractionInput.addEventListener("change", () => {
      const val = Math.max(0, Math.min(1, parseFloat(calibrationFractionInput.value) || 0.5));
      calibrationFractionInput.value = val;
      putJSON("/api/settings", { calibration_fraction: val }).catch(() => {});
    });
  }

  if (autopilotTopGreensInput) {
    autopilotTopGreensInput.addEventListener("change", () => {
      const val = Math.max(1, parseInt(autopilotTopGreensInput.value) || 10);
      autopilotTopGreensInput.value = val;
      putJSON("/api/settings", { autopilot_top_greens: val }).catch(() => {});
    });
  }

  if (autopilotHardRedsInput) {
    autopilotHardRedsInput.addEventListener("change", () => {
      const val = Math.max(1, parseInt(autopilotHardRedsInput.value) || 10);
      autopilotHardRedsInput.value = val;
      putJSON("/api/settings", { autopilot_hard_reds: val }).catch(() => {});
    });
  }

  if (settingsDefaultBtn) {
    settingsDefaultBtn.addEventListener("click", async () => {
      try {
        const res = await fetch("/api/settings/defaults");
        if (!res.ok) return;
        const defaults = await res.json();
        await putJSON("/api/settings", defaults);
        populateSettingsModal(defaults);
        applyTheme(defaults.theme || "dark");
        if (inclusionSlider) {
          inclusionSlider.value = defaults.inclusion || 0;
          inclusionValue.textContent = defaults.inclusion || 0;
          State.inclusion = defaults.inclusion || 0;
        }
        State.audioVolume = defaults.volume != null ? defaults.volume : 1.0;
        const audioEl = document.getElementById("media-audio");
        if (audioEl) audioEl.volume = State.audioVolume;
        renderMediaList();
        renderVotes();
      } catch (_) {}
    });
  }

  if (settingsImportBtn && settingsImportFile) {
    settingsImportBtn.addEventListener("click", () => settingsImportFile.click());
    settingsImportFile.addEventListener("change", async () => {
      const file = settingsImportFile.files[0];
      if (!file) return;
      try {
        const text = await file.text();
        const imported = JSON.parse(text);
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
          if (inclusionSlider) {
            inclusionSlider.value = data.inclusion || 0;
            inclusionValue.textContent = data.inclusion || 0;
            State.inclusion = data.inclusion || 0;
          }
          State.audioVolume = typeof data.volume === "number" ? data.volume : 1.0;
          const audioEl = document.getElementById("media-audio");
          if (audioEl) audioEl.volume = State.audioVolume;
          renderMediaList();
          renderVotes();
        }
      } catch (_) {
        vtAlert("Failed to import settings. Make sure the file is valid JSON.", "error");
      }
      settingsImportFile.value = "";
    });
  }

  if (settingsExportBtn) {
    settingsExportBtn.addEventListener("click", async () => {
      try {
        const res = await fetch("/api/settings");
        if (!res.ok) return;
        const data = await res.json();
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

  // Theme buttons
  themeBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      const theme = btn.dataset.theme;
      applyTheme(theme);
      saveTheme(theme);
      const themeNames = { light: "Light", dark: "Dark", highviz: "High Visibility" };
      import("./ui.js").then((ui) => ui.announce(`${themeNames[theme] || theme} mode enabled`));
    });
  });
}

export { inclusionSlider, inclusionValue, settingsModal };
