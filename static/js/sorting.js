/**
 * Sorting module — text sort, learned sort, load/detector sort,
 * sort progress polling, and sort-mode switching.
 */

import State from "./state.js";
import * as API from "./api.js";
import { formatETA } from "./ui.js";

// ---- DOM refs ----------------------------------------------------------

const sortStatus       = document.getElementById("sort-status");
const sortProgress     = document.getElementById("sort-progress");
const sortProgressFill = document.querySelector(".sort-progress-fill");
const textSortInput    = document.getElementById("text-sort");
const textSortWrap     = document.getElementById("text-sort-wrap");
const learnedSortWrap  = document.getElementById("learned-sort-wrap");
const learnedSortDesc  = document.getElementById("learned-sort-desc");
const loadSortWrap     = document.getElementById("load-sort-wrap");
const loadSortDesc     = document.getElementById("load-sort-desc");
const learnedRadio     = document.getElementById("learned-radio");
const loadRadio        = document.getElementById("load-radio");

// ---- Sort progress bar -------------------------------------------------

export function showSortProgress(label) {
  sortStatus.textContent = label;
  sortProgressFill.style.width = "";
  sortProgressFill.classList.remove("determinate");
  sortProgress.classList.add("active");
  State.sortEtaState = null;
}

export function showSortProgressWithPolling(label) {
  showSortProgress(label);
  startSortProgressPolling();
}

export function hideSortProgress() {
  stopSortProgressPolling();
  sortProgress.classList.remove("active");
  State.sortEtaState = null;
}

async function pollSortProgress() {
  try {
    const progress = await API.getSortProgress();
    if (progress.status === "idle") return;
    if (progress.total > 0) {
      const pct = Math.round((progress.current / progress.total) * 100);
      sortProgressFill.classList.add("determinate");
      sortProgressFill.style.width = `${pct}%`;

      const now = Date.now();
      if (!State.sortEtaState || State.sortEtaState.total !== progress.total) {
        State.sortEtaState = { startTime: now, startCurrent: progress.current, total: progress.total };
      }
      const elapsed = (now - State.sortEtaState.startTime) / 1000;
      const done = progress.current - State.sortEtaState.startCurrent;
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
  if (State.sortProgressTimer) return;
  State.sortProgressTimer = setInterval(pollSortProgress, 200);
}

function stopSortProgressPolling() {
  if (State.sortProgressTimer) {
    clearInterval(State.sortProgressTimer);
    State.sortProgressTimer = null;
  }
}

// ---- Text sort ---------------------------------------------------------

export async function fetchTextSort(text) {
  showSortProgressWithPolling("Searching and sorting\u2026");
  try {
    const data = await API.postTextSort(text);
    State.sortOrder = data.results.map((e) => ({ id: e.id, score: e.similarity }));
    State.threshold = data.threshold;
    hideSortProgress();
    sortStatus.textContent = `Threshold: ${(State.threshold * 100).toFixed(1)}%`;
    return true; // signal success to caller
  } catch (error) {
    hideSortProgress();
    sortStatus.textContent = `Error: ${error.message}`;
    console.error("Sort error:", error);
    return false;
  }
}

export function onTextSortInput(renderMediaList, findNextClip, selectMedia) {
  clearTimeout(State.sortTimer);
  const text = textSortInput.value.trim();
  if (!text) {
    State.sortOrder = null;
    sortStatus.textContent = "";
    renderMediaList();
    return;
  }
  State.sortTimer = setTimeout(async () => {
    const ok = await fetchTextSort(text);
    if (ok) {
      renderMediaList();
      const next = findNextClip();
      if (next) selectMedia(next.id);
    }
  }, 400);
}

// ---- Learned sort ------------------------------------------------------

export async function fetchLearnedSort(autoSelect = false) {
  showSortProgress("Training\u2026");
  try {
    const res = await fetch("/api/learned-sort", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
    });
    if (!res.ok) {
      State.sortOrder = null;
      State.threshold = null;
      hideSortProgress();
      sortStatus.textContent = "Vote good & bad first";
      return { ok: false };
    }
    const data = await res.json();
    State.sortOrder = data.results;
    State.threshold = data.threshold;
    const fgScores = {};
    data.results.forEach((r) => { fgScores[String(r.id)] = r.score; });
    State.votes.learned_scores = fgScores;
    hideSortProgress();
    sortStatus.textContent = `Threshold: ${(State.threshold * 100).toFixed(1)}%`;
    return { ok: true, autoSelect };
  } catch (error) {
    hideSortProgress();
    sortStatus.textContent = `Error: ${error.message}`;
    console.error("Learned sort error:", error);
    return { ok: false };
  }
}

export function scheduleLearnedSort(renderMediaList, renderVotes, delay = 300) {
  clearTimeout(State.learnedSortDebounce);
  State.learnedSortDebounce = setTimeout(() => {
    if (State.learnedSortController) {
      State.learnedSortController.abort();
    }
    State.learnedSortController = new AbortController();
    const controller = State.learnedSortController;

    showSortProgress("Training\u2026");

    fetch("/api/learned-sort", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: controller.signal,
    })
      .then((res) => {
        if (!res.ok) {
          State.sortOrder = null;
          State.threshold = null;
          hideSortProgress();
          sortStatus.textContent = "Vote good & bad first";
          renderMediaList();
          return null;
        }
        return res.json();
      })
      .then((data) => {
        if (!data) return;
        State.sortOrder = data.results;
        State.threshold = data.threshold;
        const newScores = {};
        data.results.forEach((r) => { newScores[String(r.id)] = r.score; });
        State.votes.learned_scores = newScores;
        hideSortProgress();
        sortStatus.textContent = `Threshold: ${(State.threshold * 100).toFixed(1)}%`;
        renderMediaList();
        renderVotes();
      })
      .catch((err) => {
        if (err.name === "AbortError") return;
        hideSortProgress();
        sortStatus.textContent = `Error: ${err.message}`;
        console.error("Learned sort error:", err);
      });
  }, delay);
}

// ---- Loaded (detector) sort --------------------------------------------

export async function fetchLoadedSort(autoSelect = false) {
  if (!State.loadedDetector) {
    sortStatus.textContent = "Load a sort first";
    return { ok: false };
  }
  if (State.loadedDetector._example) {
    if (State.sortOrder && State.threshold != null) {
      sortStatus.textContent = `Threshold: ${(State.threshold * 100).toFixed(1)}%`;
      return { ok: true, autoSelect };
    }
    return { ok: false };
  }
  showSortProgress("Scoring with loaded detector\u2026");
  try {
    const data = await API.postDetectorSort(State.loadedDetector);
    State.sortOrder = data.results;
    State.threshold = data.threshold;
    hideSortProgress();
    sortStatus.textContent = `Threshold: ${(State.threshold * 100).toFixed(1)}%`;
    return { ok: true, autoSelect };
  } catch (error) {
    hideSortProgress();
    sortStatus.textContent = `Error: ${error.message}`;
    console.error("Detector sort error:", error);
    return { ok: false };
  }
}

// ---- Sort-mode UI switching helpers ------------------------------------

export function updateSortModeAvailability() {
  const hasGoodAndBad = State.votes.good.length > 0 && State.votes.bad.length > 0;
  learnedRadio.disabled = !hasGoodAndBad;
  learnedRadio.parentElement.style.opacity = hasGoodAndBad ? "1" : "0.5";
  learnedRadio.parentElement.style.cursor = hasGoodAndBad ? "pointer" : "not-allowed";
  loadRadio.disabled = false;
  loadRadio.parentElement.style.opacity = "1";
  loadRadio.parentElement.style.cursor = "pointer";
}

export function updateLearnedSortDesc() {
  if (!learnedSortDesc) return;
  const nGood = State.votes.good.length;
  const nBad = State.votes.bad.length;
  learnedSortDesc.textContent = `Training on ${nGood} good + ${nBad} bad`;
}

export function activateLoadSort(label) {
  State.sortMode = "load";
  document.querySelectorAll('input[name="sort-mode"]').forEach((r) => {
    r.checked = r.value === "load";
  });
  textSortWrap.style.display = "none";
  learnedSortWrap.style.display = "none";
  loadSortWrap.style.display = "";
  if (label && loadSortDesc) loadSortDesc.textContent = label;
}

export function getTextSortInput() {
  return textSortInput;
}

export function setSortStatus(text) {
  sortStatus.textContent = text;
}

export { textSortWrap, learnedSortWrap, loadSortWrap, sortStatus };
