/**
 * Voting module — castVote, vote rendering, label sorting, stripe overview.
 */

import State from "./state.js";
import * as API from "./api.js";
import { escapeHtml, announce } from "./ui.js";

// ---- DOM refs ----------------------------------------------------------

const goodList      = document.getElementById("good-list");
const badList       = document.getElementById("bad-list");
const stripeContainer = document.getElementById("stripe-container");
const stripeOverview  = document.getElementById("stripe-overview");

// ---- Label counts ------------------------------------------------------

export function updateLabelCounts() {
  const goodCount = document.getElementById("good-count");
  const badCount = document.getElementById("bad-count");
  if (goodCount) goodCount.textContent = `(${State.votes.good.length})`;
  if (badCount) badCount.textContent = `(${State.votes.bad.length})`;
}

// ---- Stripe overview ---------------------------------------------------

export function renderStripe() {
  if (!stripeContainer) return;
  stripeContainer.innerHTML = "";

  const ordered = State.sortOrder
    ? State.sortOrder.map((s) => State.medias.find((c) => c.id === s.id)).filter(Boolean)
    : State.medias;

  ordered.forEach((c) => {
    const s = document.createElement("div");
    s.className = "stripe-cell";
    if (State.votes.good.includes(c.id)) s.classList.add("good");
    else if (State.votes.bad.includes(c.id)) s.classList.add("bad");
    if (c.id === State.selected) s.classList.add("selected");
    stripeContainer.appendChild(s);
  });
}

// ---- Label sort helpers ------------------------------------------------

function labelSortKey(id, label) {
  const media = State.medias.find((c) => c.id === id);
  const name = media ? (media.filename || `Clip #${id}`) : `Clip #${id}`;
  const time = State.votes.click_times[String(id)] ?? -1;
  const score = State.votes.learned_scores[String(id)] ?? -1;
  let confidence = -1;
  if (score >= 0) {
    confidence = label === "good" ? score : 1 - score;
  }
  return { id, name, time, score, confidence };
}

function sortLabelEntries(ids, label) {
  const entries = ids.map((id) => labelSortKey(id, label));
  switch (State.labelSortMode) {
    case "time-desc":
      entries.sort((a, b) => b.time - a.time); break;
    case "time-asc":
      entries.sort((a, b) => a.time - b.time); break;
    case "name-asc":
      entries.sort((a, b) => a.name.localeCompare(b.name)); break;
    case "name-desc":
      entries.sort((a, b) => b.name.localeCompare(a.name)); break;
    case "confidence-desc":
      entries.sort((a, b) => b.confidence - a.confidence); break;
    case "confidence-asc":
      entries.sort((a, b) => a.confidence - b.confidence); break;
    case "id-asc":
      entries.sort((a, b) => a.id - b.id); break;
    default:
      entries.sort((a, b) => b.time - a.time);
  }
  return entries;
}

function mediaSupportsThumbnail(media) {
  return media && (media.type === "image" || media.type === "video");
}

function thumbnailUrl(media) {
  if (media.type === "image") return `/api/medias/${media.id}/image`;
  if (media.type === "video") return `/api/medias/${media.id}/video`;
  return "";
}

// ---- Render votes (right panel) ----------------------------------------

export function renderVotes(selectMediaFn) {
  if (!goodList || !badList) return;

  function renderVoteList(listEl, ids, label) {
    listEl.innerHTML = "";
    const entries = sortLabelEntries(ids, label);
    entries.forEach((entry) => {
      const div = document.createElement("div");
      div.className = "vote-item";
      div.setAttribute("role", "button");
      div.setAttribute("tabindex", "0");
      if (entry.id === State.selected) div.classList.add("active");

      const media = State.medias.find((c) => c.id === entry.id);
      let html = "";
      const metaParts = [];

      if (entry.confidence >= 0) {
        metaParts.push(`${(entry.confidence * 100).toFixed(0)}%`);
      }

      if (State.showThumbnailsRight && media && mediaSupportsThumbnail(media)) {
        div.classList.add("vote-item-thumb");
        html += `<img class="vote-thumbnail" src="${thumbnailUrl(media)}" alt="${escapeHtml(entry.name)}" loading="lazy">`;
      }

      html += `<span class="vote-name">${escapeHtml(entry.name)}</span><span class="vote-meta">${metaParts.join(" \u00b7 ")}</span>`;
      div.innerHTML = html;

      div.addEventListener("click", () => selectMediaFn(entry.id));
      div.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") { e.preventDefault(); selectMediaFn(entry.id); }
      });
      listEl.appendChild(div);
    });
  }

  renderVoteList(goodList, State.votes.good, "good");
  renderVoteList(badList, State.votes.bad, "bad");
  updateLabelCounts();
}

// ---- Fetch votes -------------------------------------------------------

export async function fetchVotes(renderCallbacks) {
  const data = await API.getVotes();
  State.votes = data;
  renderCallbacks.renderVotes();
  renderStripe();
  renderCallbacks.updateSortModeAvailability();
  if (State.selected) renderCallbacks.renderCenter();
}

// ---- Cast vote ---------------------------------------------------------

export async function castVote(id, vote, callbacks) {
  if (State.isVoting) return;
  State.isVoting = true;
  try {
    const btnId = vote === "good" ? "vote-good" : "vote-bad";
    const clickedBtn = document.getElementById(btnId);
    const wasVoted = clickedBtn && clickedBtn.classList.contains("voted");
    if (clickedBtn) {
      if (wasVoted) clickedBtn.classList.remove("voted");
      else clickedBtn.classList.add("vote-flash");
    }

    const mediaName = (State.medias.find((c) => c.id === id) || {}).filename || `Clip #${id}`;
    await API.postVote(id, vote);
    announce(`Voted ${vote} on ${mediaName}`);
    await fetchVotes(callbacks);

    if (State._dashboardTrainMode) {
      callbacks.persistTrainableModelLabels();
    }

    if (vote === "good" && State.sortMode === "text") {
      const textQuery = callbacks.getTextSortValue();
      if (textQuery) {
        API.postTextsortSuggestion(textQuery);
      }
    }

    callbacks.autopilotCountHardLabel();
    callbacks.checkAutopilotPhase();

    let nextId;
    if (State.selectMode === "new") {
      const data = await API.postDiversityTreeNext(
        State.sortOrder && State.sortOrder.length > 0
          ? { scores: Object.fromEntries(State.sortOrder.map((e) => [e.id, e.score ?? e.similarity ?? 0])) }
          : {},
      );
      nextId = data.id;
      if (nextId == null && data.exhausted) {
        const { vtAlert } = await import("./ui.js");
        vtAlert("You have seen every branch of the diversity tree. Switch to Top or Hard mode, or add more data.", "warning");
      }
    } else {
      const c = callbacks.findNextClip();
      nextId = c ? c.id : null;
    }

    if (nextId != null && nextId !== State.selected) {
      if (State.swipeAnimation) {
        const dir = vote === "good" ? "swipe-right" : "swipe-left";
        const wrapper = document.getElementById("media-swipe-wrapper");
        if (wrapper) {
          wrapper.classList.add(dir);
          await new Promise((r) => setTimeout(r, 180));
          wrapper.classList.remove(dir);
        }
      }
      callbacks.selectMedia(nextId);
    } else {
      callbacks.renderMediaList();
      callbacks.renderCenter();
    }

    if (State.sortMode === "learned") {
      callbacks.scheduleLearnedSort();
    }
  } finally {
    State.isVoting = false;
  }
}

export { thumbnailUrl, mediaSupportsThumbnail };
