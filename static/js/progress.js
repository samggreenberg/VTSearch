/**
 * Progress & labeling-status module — indicator buttons, labeling analysis
 * modal, progress charts, and dataset-loading progress bars.
 */

import State from "./state.js";
import { themeColor, vtAlert, formatETA } from "./ui.js";

// ---- DOM refs ----------------------------------------------------------

const smartIndicator        = document.getElementById("smart-indicator");
const stableIndicator       = document.getElementById("stable-indicator");
const spanIndicator         = document.getElementById("span-indicator");
const progressModal         = document.getElementById("progress-modal");
const modalClose            = document.getElementById("modal-close");
const labelingAnalysisModal = document.getElementById("labeling-analysis-modal");
const labelingAnalysisBar   = document.getElementById("labeling-analysis-bar");
const labelingAnalysisText  = document.getElementById("labeling-analysis-text");
const labelingAnalysisPct   = document.getElementById("labeling-analysis-pct");

// Dataset loading progress elements
const datasetProgress  = document.getElementById("dataset-progress");
const progressFill     = document.getElementById("progress-fill");
const progressText     = document.getElementById("progress-text");
const progressMessage  = document.getElementById("progress-message");
const progressEta      = document.getElementById("progress-eta");

let _lastStatusData = null;
let _statusTimer = null;

// ---- Labeling status indicators ----------------------------------------

function _applyIndicator(btn, subtextEl, metric) {
  btn.dataset.status = metric.status;
  if (subtextEl) subtextEl.textContent = "";
}

export function applyLabelingStatus(data, autopilotCallback) {
  _lastStatusData = data;
  if (data.smart) _applyIndicator(smartIndicator, document.getElementById("smart-subtext"), data.smart);
  if (data.stable) _applyIndicator(stableIndicator, document.getElementById("stable-subtext"), data.stable);
  if (data.span) _applyIndicator(spanIndicator, document.getElementById("span-subtext"), data.span);
  if (autopilotCallback) autopilotCallback(data);
}

export function scheduleLabelingStatusUpdate() {
  clearTimeout(_statusTimer);
  _statusTimer = setTimeout(fetchLabelingStatus, 1200);
}

let _autopilotIndicatorCb = null;
export function setAutopilotIndicatorCallback(cb) { _autopilotIndicatorCb = cb; }

export async function fetchLabelingStatus() {
  try {
    const res = await fetch("/api/labeling-status");
    if (!res.ok) return;
    const data = await res.json();
    if (data.error) return;
    applyLabelingStatus(data, _autopilotIndicatorCb);
  } catch (_) {}
}

// ---- Metric detail / progress analysis modal ---------------------------

export function initProgressIndicators(callbacks) {
  const { pauseActiveMedia, resumeActiveMedia } = callbacks;

  smartIndicator.addEventListener("click", () => showMetricDetail("smart", smartIndicator, callbacks));
  stableIndicator.addEventListener("click", () => showMetricDetail("stable", stableIndicator, callbacks));
  spanIndicator.addEventListener("click", () => showMetricDetail("span", spanIndicator, callbacks));

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
}

async function showMetricDetail(metric, triggerBtn, callbacks) {
  const { pauseActiveMedia, resumeActiveMedia } = callbacks;

  if (metric === "span" && (State.votes.good.length === 0 || State.votes.bad.length === 0)) {
    showSpanPopup(pauseActiveMedia);
    return;
  }

  if (State.votes.good.length === 0 || State.votes.bad.length === 0) {
    await vtAlert("Need at least one good and one bad vote to check progress", "warning");
    return;
  }

  pauseActiveMedia();

  labelingAnalysisBar.style.width = "0%";
  labelingAnalysisPct.textContent = "0%";
  labelingAnalysisText.textContent = "Training models over label history\u2026";
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
  triggerBtn.querySelector(".indicator-label").textContent = "\u2026";

  try {
    const res = await fetch("/api/labeling-progress", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
    });

    clearInterval(progressInterval);
    labelingAnalysisBar.style.width = "100%";
    labelingAnalysisPct.textContent = "100%";
    labelingAnalysisText.textContent = "Done!";

    await new Promise((r) => setTimeout(r, 350));
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

function showSpanPopup(pauseActiveMedia) {
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

  if (_lastStatusData) {
    document.getElementById("stat-total-labels").textContent = _lastStatusData.total_count || 0;
    document.getElementById("stat-total-medias").textContent = "\u2014";
    const currentType = State.medias.length > 0 ? State.medias[0].type : null;
    const mtInfo = currentType ? State.mediaTypesMap[currentType] : null;
    document.getElementById("stat-total-medias-label").textContent = mtInfo ? `Total ${mtInfo.tab_title}` : "Total Medias";
  }

  document.getElementById("smart-section").style.display = "none";
  document.getElementById("stable-section").style.display = "none";
  document.getElementById("span-section").style.display = "";
  document.getElementById("progress-modal-title").textContent = "Diverse: Diversity Coverage";

  pauseActiveMedia();
  progressModal.classList.add("show");
}

function displayProgressResults(data, metric) {
  document.getElementById("stat-total-labels").textContent = data.total_labels;
  document.getElementById("stat-total-medias").textContent = data.total_medias;
  const currentType = State.medias.length > 0 ? State.medias[0].type : null;
  const mtInfo = currentType ? State.mediaTypesMap[currentType] : null;
  document.getElementById("stat-total-medias-label").textContent = mtInfo ? `Total ${mtInfo.tab_title}` : "Total Medias";

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
  } else if (metric === "stable") {
    stableSec.style.display = "";
    renderStabilityChart(data.stability_over_time);
    document.getElementById("progress-modal-title").textContent = "Stable: Prediction Flip Analysis";
  } else if (metric === "span") {
    spanSec.style.display = "";
    renderDiversityChart(data.diversity_level_over_time);
    document.getElementById("progress-modal-title").textContent = "Diverse: Diversity Coverage";
    updateSpanInfoText();
  }
}

function updateSpanInfoText() {
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

// ---- Charts ------------------------------------------------------------

function renderErrorCostChart(errorCostData) {
  const canvas = document.getElementById("error-cost-chart");
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (!errorCostData || errorCostData.length === 0) {
    ctx.fillStyle = themeColor("--text-muted");
    ctx.font = "14px sans-serif";
    ctx.fillText("No data available", 20, canvas.height / 2);
    return;
  }

  const numLabels = errorCostData.map((d) => d.num_labels);
  const errorCosts = errorCostData.map((d) => d.error_cost);
  const padding = { top: 20, right: 20, bottom: 40, left: 50 };
  const chartWidth = canvas.width - padding.left - padding.right;
  const chartHeight = canvas.height - padding.top - padding.bottom;
  const maxLabels = Math.max(...numLabels);
  const maxCost = Math.max(...errorCosts);
  const minCost = Math.min(...errorCosts);
  const xScale = (val) => padding.left + (val / maxLabels) * chartWidth;
  const yScale = (val) => padding.top + chartHeight - ((val - minCost) / (maxCost - minCost || 1)) * chartHeight;

  // Axes
  ctx.strokeStyle = themeColor("--border");
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(padding.left, padding.top);
  ctx.lineTo(padding.left, padding.top + chartHeight);
  ctx.lineTo(padding.left + chartWidth, padding.top + chartHeight);
  ctx.stroke();

  // Line
  ctx.strokeStyle = themeColor("--accent");
  ctx.lineWidth = 2;
  ctx.beginPath();
  errorCostData.forEach((d, i) => {
    const x = xScale(d.num_labels);
    const y = yScale(d.error_cost);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Labels
  ctx.fillStyle = themeColor("--text-secondary");
  ctx.font = "11px sans-serif";
  ctx.textAlign = "center";
  ctx.fillText("# Labels", padding.left + chartWidth / 2, canvas.height - 5);
  ctx.save();
  ctx.translate(12, padding.top + chartHeight / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("Error Cost", 0, 0);
  ctx.restore();
}

function renderStabilityChart(stabilityData) {
  const canvas = document.getElementById("stability-chart");
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (!stabilityData || stabilityData.length < 2) {
    ctx.fillStyle = themeColor("--text-muted");
    ctx.font = "14px sans-serif";
    ctx.fillText("Not enough data yet", 20, canvas.height / 2);
    return;
  }

  const numLabels = stabilityData.map((d) => d.num_labels);
  const flips = stabilityData.map((d) => d.num_flips);
  const padding = { top: 20, right: 20, bottom: 40, left: 50 };
  const chartWidth = canvas.width - padding.left - padding.right;
  const chartHeight = canvas.height - padding.top - padding.bottom;
  const maxLabels = Math.max(...numLabels);
  const maxFlips = Math.max(...flips);
  const xScale = (val) => padding.left + (val / maxLabels) * chartWidth;
  const yScale = (val) => padding.top + chartHeight - (val / (maxFlips || 1)) * chartHeight;

  ctx.strokeStyle = themeColor("--border");
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(padding.left, padding.top);
  ctx.lineTo(padding.left, padding.top + chartHeight);
  ctx.lineTo(padding.left + chartWidth, padding.top + chartHeight);
  ctx.stroke();

  ctx.strokeStyle = themeColor("--accent");
  ctx.lineWidth = 2;
  ctx.beginPath();
  stabilityData.forEach((d, i) => {
    const x = xScale(d.num_labels);
    const y = yScale(d.num_flips);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  ctx.fillStyle = themeColor("--text-secondary");
  ctx.font = "11px sans-serif";
  ctx.textAlign = "center";
  ctx.fillText("# Labels", padding.left + chartWidth / 2, canvas.height - 5);
  ctx.save();
  ctx.translate(12, padding.top + chartHeight / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("Prediction Flips", 0, 0);
  ctx.restore();
}

function renderDiversityChart(diversityData) {
  const canvas = document.getElementById("diversity-chart");
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (!diversityData || diversityData.length === 0) {
    ctx.fillStyle = themeColor("--text-muted");
    ctx.font = "14px sans-serif";
    ctx.fillText("No diversity data yet", 20, canvas.height / 2);
    return;
  }

  const numLabels = diversityData.map((d) => d.num_labels);
  const levels = diversityData.map((d) => d.diversity_level);
  const padding = { top: 20, right: 20, bottom: 40, left: 50 };
  const chartWidth = canvas.width - padding.left - padding.right;
  const chartHeight = canvas.height - padding.top - padding.bottom;
  const maxLabels = Math.max(...numLabels);
  const maxLevel = Math.max(...levels, 1);
  const xScale = (val) => padding.left + (val / maxLabels) * chartWidth;
  const yScale = (val) => padding.top + chartHeight - (val / maxLevel) * chartHeight;

  ctx.strokeStyle = themeColor("--border");
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(padding.left, padding.top);
  ctx.lineTo(padding.left, padding.top + chartHeight);
  ctx.lineTo(padding.left + chartWidth, padding.top + chartHeight);
  ctx.stroke();

  ctx.strokeStyle = "#2ecc71";
  ctx.lineWidth = 2;
  ctx.beginPath();
  diversityData.forEach((d, i) => {
    const x = xScale(d.num_labels);
    const y = yScale(d.diversity_level);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  ctx.fillStyle = themeColor("--text-secondary");
  ctx.font = "11px sans-serif";
  ctx.textAlign = "center";
  ctx.fillText("# Labels", padding.left + chartWidth / 2, canvas.height - 5);
  ctx.save();
  ctx.translate(12, padding.top + chartHeight / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("Diversity Level", 0, 0);
  ctx.restore();
}

// ---- Dataset-loading progress (welcome screen) -------------------------

export function showDatasetProgress() {
  datasetProgress.style.display = "block";
  progressFill.style.width = "0%";
  progressFill.classList.add("indeterminate");
  progressText.textContent = "";
  progressMessage.textContent = "Loading...";
  progressMessage.style.color = "";
  if (progressEta) progressEta.textContent = "";
}

export function pollDatasetProgress(callbacks) {
  const { onComplete, onError } = callbacks;

  async function poll() {
    try {
      const res = await fetch("/api/dataset/progress");
      const progress = await res.json();

      if (progress.error) {
        stopDatasetProgressPolling();
        if (progressMessage) {
          progressMessage.textContent = `Error: ${progress.error}`;
          progressMessage.style.color = "var(--color-bad)";
        }
        progressFill.classList.remove("indeterminate");
        if (onError) onError(progress.error);
        return;
      }

      if (progress.pct != null) {
        progressFill.classList.remove("indeterminate");
        progressFill.style.width = `${progress.pct}%`;
        progressText.textContent = `${progress.pct}%`;

        // ETA calculation
        const now = Date.now();
        if (!State.progressEtaState || State.progressEtaState.total !== progress.total) {
          State.progressEtaState = { startTime: now, startCurrent: progress.current, total: progress.total };
        }
        if (progress.current > 0 && progressEta) {
          const elapsed = (now - State.progressEtaState.startTime) / 1000;
          const done = progress.current - State.progressEtaState.startCurrent;
          if (done > 0 && elapsed > 2) {
            const rate = done / elapsed;
            const remaining = (progress.total - progress.current) / rate;
            progressEta.textContent = formatETA(remaining);
          }
        }
      }
      if (progress.message) {
        progressMessage.textContent = progress.message;
        progressMessage.style.color = "";
      }
      if (progress.status === "idle") {
        stopDatasetProgressPolling();
        State.progressEtaState = null;
        if (onComplete) onComplete();
      }
    } catch (_) {}
  }

  startDatasetProgressPolling(poll);
}

let _datasetProgressTimer = null;

function startDatasetProgressPolling(pollFn) {
  if (_datasetProgressTimer) clearInterval(_datasetProgressTimer);
  _datasetProgressTimer = setInterval(pollFn, 1000);
}

export function stopDatasetProgressPolling() {
  if (_datasetProgressTimer) {
    clearInterval(_datasetProgressTimer);
    _datasetProgressTimer = null;
  }
}

export { progressModal };
