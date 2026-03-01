/**
 * Media module — rendering the media list, center panel player,
 * waveform drawing, image view controls, and next-clip selection.
 */

import State from "./state.js";
import { escapeHtml, themeColor, announce } from "./ui.js";
import { mediaSupportsThumbnail, thumbnailUrl } from "./voting.js";

// ---- DOM refs ----------------------------------------------------------

const mediaList = document.getElementById("media-list");
const center    = document.getElementById("center");
const stripeContainer = document.getElementById("stripe-container");

// ---- Next-clip selection -----------------------------------------------

export function findNextClip() {
  if (State.selectMode === "new") return null;

  let ordered = State.sortOrder;
  let effectiveThreshold = State.threshold;
  if (!ordered || ordered.length === 0) {
    ordered = State.medias.map((c) => ({ id: c.id, score: 0 }));
    effectiveThreshold = -Infinity;
  }
  if (ordered.length === 0) return null;

  const unlabeled = ordered.filter(
    (item) => !State.votes.good.includes(item.id) && !State.votes.bad.includes(item.id),
  );
  if (unlabeled.length === 0) return null;

  if (State.selectMode === "top") return unlabeled[0];

  // Hard mode — closest to threshold
  if (effectiveThreshold === null) return null;
  let thresholdIdx = ordered.length;
  for (let i = 0; i < ordered.length; i++) {
    if (ordered[i].score < effectiveThreshold) { thresholdIdx = i; break; }
  }

  const idToIdx = {};
  ordered.forEach((item, idx) => { idToIdx[item.id] = idx; });

  let minIdxDist = Infinity;
  let minDist = Infinity;
  let nextClip = null;
  for (const item of unlabeled) {
    const idxDist = Math.abs(idToIdx[item.id] - thresholdIdx);
    const dist = Math.abs(item.score - effectiveThreshold);
    if (idxDist < minIdxDist || (idxDist === minIdxDist && dist < minDist)) {
      minIdxDist = idxDist;
      minDist = dist;
      nextClip = item;
    }
  }
  return nextClip;
}

// ---- Fetch & render media list -----------------------------------------

export async function fetchMedias() {
  const data = await fetch("/api/medias").then((r) => r.json());
  State.medias = data;
}

export function renderMediaList(selectMediaFn) {
  mediaList.innerHTML = "";
  const scoreMap = {};
  if (State.sortOrder) {
    State.sortOrder.forEach((s) => { scoreMap[s.id] = s.score; });
  }

  const ordered = State.sortOrder
    ? State.sortOrder.map((s) => State.medias.find((c) => c.id === s.id)).filter(Boolean)
    : State.medias;

  let thresholdLineInserted = false;
  ordered.forEach((c) => {
    if (State.threshold !== null && !thresholdLineInserted && scoreMap[c.id] !== undefined && scoreMap[c.id] < State.threshold) {
      const line = document.createElement("div");
      line.className = "media-threshold-line";
      mediaList.appendChild(line);
      thresholdLineInserted = true;
    }

    const div = document.createElement("div");
    const isGood = State.votes.good.includes(c.id);
    const isBad = State.votes.bad.includes(c.id);
    let className = "media-item";
    if (State.selected === c.id) className += " active";
    if (isGood) className += " labeled-good";
    if (isBad) className += " labeled-bad";
    div.className = className;
    div.setAttribute("role", "option");
    div.setAttribute("tabindex", "0");
    div.setAttribute("aria-selected", State.selected === c.id ? "true" : "false");
    const mediaLabel = c.filename || "Media #" + c.id;
    const labelParts = [mediaLabel];
    if (isGood) labelParts.push("labeled good");
    if (isBad) labelParts.push("labeled bad");
    if (scoreMap[c.id] !== undefined) labelParts.push(`score ${(scoreMap[c.id] * 100).toFixed(1)}%`);
    div.setAttribute("aria-label", labelParts.join(", "));

    let html = "";
    const useThumbnail = State.showThumbnailsLeft && mediaSupportsThumbnail(c);
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
    div.onclick = () => selectMediaFn(c.id);
    div.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); selectMediaFn(c.id); }
    });
    mediaList.appendChild(div);
  });
}

// ---- Select media ------------------------------------------------------

export function selectMedia(id, callbacks) {
  State.selected = id;
  callbacks.renderMediaList();
  callbacks.renderCenter();

  const activeItem = mediaList.querySelector(".media-item.active");
  if (activeItem) {
    activeItem.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }

  const c = State.medias.find((x) => x.id === id);
  if (c) announce(`Selected ${c.filename || "Media #" + c.id}`);
}

// ---- Center panel rendering --------------------------------------------

export function renderCenter(castVoteFn, saveVolumeFn) {
  const c = State.medias.find((x) => x.id === State.selected);
  if (!c) return;
  const isGood = State.votes.good.includes(c.id);
  const isBad = State.votes.bad.includes(c.id);
  center.className = "panel-center";

  const mediaType = c.type || "audio";
  const mtInfo = State.mediaTypesMap[mediaType];

  let playerHTML = "";
  if (mediaType === "video") {
    playerHTML = `<video controls loop autoplay src="/api/medias/${c.id}/video" id="media-video" aria-label="${escapeHtml(c.filename || "Video media")}" class="media-player-video"></video>`;
  } else if (mediaType === "image") {
    playerHTML = `<div class="media-player-image-wrap"><img src="/api/medias/${c.id}/image" id="media-image" alt="${escapeHtml(c.filename || "Image media")}" class="media-player-image"></div>`;
  } else if (mediaType === "paragraph") {
    playerHTML = `<div id="media-paragraph" class="media-player-text">Loading...</div>`;
  } else if (mediaType === "audio") {
    playerHTML = `<canvas id="waveform-canvas" width="600" height="120" role="img" aria-label="Audio waveform visualization"></canvas>`;
  } else {
    const loops = mtInfo && mtInfo.loops;
    if (loops) {
      playerHTML = `<video controls loop autoplay src="/api/medias/${c.id}/media" id="media-video" class="media-player-video"></video>`;
    } else {
      playerHTML = `<div class="media-player-image-wrap"><object data="/api/medias/${c.id}/media" class="media-player-embed">${escapeHtml(c.filename || "Media")}</object></div>`;
    }
  }

  center.innerHTML = `
    <div class="media-swipe-wrapper" id="media-swipe-wrapper">
      ${playerHTML}
    </div>
    ${mediaType === "audio" ? `<audio controls controlslist="nodownload" loop autoplay src="/api/medias/${c.id}/audio" id="media-audio" aria-label="${escapeHtml(c.filename || "Audio media")}"></audio>` : ""}
    ${mediaType === "image" ? `
    <div class="image-view-controls" id="image-view-controls">
      <button class="ivc-btn" id="ivc-rotate-left" title="Rotate left" aria-label="Rotate image left">&#x21BA;</button>
      <button class="ivc-btn" id="ivc-rotate-right" title="Rotate right" aria-label="Rotate image right">&#x21BB;</button>
      <label for="ivc-zoom" class="sr-only">Zoom</label>
      <input type="range" id="ivc-zoom" class="ivc-zoom-slider" min="0.25" max="5" step="0.05" value="1" title="Zoom" aria-label="Zoom level">
      <span class="ivc-zoom-label" id="ivc-zoom-label">1\u00D7</span>
      <button class="ivc-btn" id="ivc-reset" title="Reset view" aria-label="Reset image view">Reset</button>
    </div>` : ""}
    <div class="metadata-grid">
      <div class="metadata-item">
        <span class="metadata-label">Name</span>
        <span class="metadata-value">${escapeHtml(c.filename || "Media #" + c.id)}</span>
      </div>
      ${c.frequency ? `
      <div class="metadata-item">
        <span class="metadata-label">Frequency</span>
        <span class="metadata-value">${c.frequency} Hz</span>
      </div>` : ""}
      ${c.category && c.category !== "unknown" ? `
      <div class="metadata-item">
        <span class="metadata-label">Category</span>
        <span class="metadata-value">${escapeHtml(c.category)}</span>
      </div>` : ""}
      <div class="metadata-item">
        <span class="metadata-label">Media Type</span>
        <span class="metadata-value">${mtInfo ? escapeHtml(mtInfo.name) : escapeHtml(mediaType)}</span>
      </div>
      ${c.duration && c.duration > 0 ? `
      <div class="metadata-item">
        <span class="metadata-label">Duration</span>
        <span class="metadata-value">${c.duration.toFixed(1)}s</span>
      </div>` : ""}
      ${c.width && c.height ? `
      <div class="metadata-item">
        <span class="metadata-label">Dimensions</span>
        <span class="metadata-value">${c.width}\u00D7${c.height}</span>
      </div>` : ""}
      ${c.word_count ? `
      <div class="metadata-item">
        <span class="metadata-label">Word Count</span>
        <span class="metadata-value">${c.word_count}</span>
      </div>
      <div class="metadata-item">
        <span class="metadata-label">Characters</span>
        <span class="metadata-value">${c.character_count}</span>
      </div>` : ""}
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

  document.getElementById("vote-good").onclick = () => castVoteFn(c.id, "good");
  document.getElementById("vote-bad").onclick = () => castVoteFn(c.id, "bad");

  if (mediaType === "audio") {
    drawWaveform(c.id);
    const audioEl = document.getElementById("media-audio");
    if (audioEl) {
      audioEl.volume = State.audioVolume;
      audioEl.addEventListener("volumechange", () => {
        State.audioVolume = audioEl.volume;
        saveVolumeFn(audioEl.volume);
      });
    }
  }

  if (mediaType === "paragraph") {
    if (State.paragraphController) State.paragraphController.abort();
    State.paragraphController = new AbortController();
    const expectedId = c.id;
    fetch(`/api/medias/${c.id}/paragraph`, { signal: State.paragraphController.signal })
      .then((res) => res.json())
      .then((data) => {
        if (State.selected !== expectedId) return;
        const paragraphDiv = document.getElementById("media-paragraph");
        if (paragraphDiv) paragraphDiv.textContent = data.content;
      })
      .catch((err) => {
        if (err.name === "AbortError") return;
        console.error("Error loading paragraph:", err);
      });
  }

  if (mediaType === "image") {
    setupImageViewControls();
  }
}

// ---- Image view controls -----------------------------------------------

function setupImageViewControls() {
  const img = document.getElementById("media-image");
  const wrap = img ? img.closest(".media-player-image-wrap") : null;
  const zoomSlider = document.getElementById("ivc-zoom");
  const zoomLabel = document.getElementById("ivc-zoom-label");
  const rotateLeftBtn = document.getElementById("ivc-rotate-left");
  const rotateRightBtn = document.getElementById("ivc-rotate-right");
  const resetBtn = document.getElementById("ivc-reset");
  if (!img || !zoomSlider || !wrap) return;

  let ivcZoom = 1, ivcRotation = 0, ivcPanX = 0, ivcPanY = 0;

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
    if (imgAspect > wrapAspect) { rendW = wrapW; rendH = wrapW / imgAspect; }
    else { rendH = wrapH; rendW = wrapH * imgAspect; }
    const rot = ((ivcRotation % 360) + 360) % 360;
    const swapped = rot === 90 || rot === 270;
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
    zoomLabel.textContent = ivcZoom.toFixed(1) + "\u00D7";
    wrap.style.cursor = max.x > 0 || max.y > 0 ? "grab" : "";
  };

  const clampZoom = (val) => Math.min(parseFloat(zoomSlider.max), Math.max(parseFloat(zoomSlider.min), val));

  zoomSlider.addEventListener("input", () => { ivcZoom = parseFloat(zoomSlider.value); applyTransform(); });
  rotateLeftBtn.addEventListener("click", () => { ivcRotation -= 90; applyTransform(); });
  rotateRightBtn.addEventListener("click", () => { ivcRotation += 90; applyTransform(); });
  resetBtn.addEventListener("click", () => {
    ivcZoom = 1; ivcRotation = 0; ivcPanX = 0; ivcPanY = 0;
    zoomSlider.value = 1;
    applyTransform();
  });

  wrap.addEventListener("wheel", (e) => {
    e.preventDefault();
    const oldZoom = ivcZoom;
    const delta = e.deltaY > 0 ? -0.15 : 0.15;
    ivcZoom = clampZoom(ivcZoom + delta * ivcZoom);
    zoomSlider.value = ivcZoom;
    const rect = wrap.getBoundingClientRect();
    const cx = e.clientX - rect.left - rect.width / 2;
    const cy = e.clientY - rect.top - rect.height / 2;
    const ratio = ivcZoom / oldZoom;
    ivcPanX = cx - ratio * (cx - ivcPanX);
    ivcPanY = cy - ratio * (cy - ivcPanY);
    applyTransform();
  }, { passive: false });

  let isPanning = false, panStartX = 0, panStartY = 0, panOriginX = 0, panOriginY = 0;
  wrap.addEventListener("mousedown", (e) => {
    const max = getMaxPan();
    if ((max.x <= 0 && max.y <= 0) || e.button !== 0) return;
    isPanning = true;
    panStartX = e.clientX; panStartY = e.clientY;
    panOriginX = ivcPanX; panOriginY = ivcPanY;
    wrap.style.cursor = "grabbing";
    e.preventDefault();
  });

  if (State._ivcWindowMoveHandler) window.removeEventListener("mousemove", State._ivcWindowMoveHandler);
  if (State._ivcWindowUpHandler) window.removeEventListener("mouseup", State._ivcWindowUpHandler);
  State._ivcWindowMoveHandler = (e) => {
    if (!isPanning) return;
    ivcPanX = panOriginX + (e.clientX - panStartX);
    ivcPanY = panOriginY + (e.clientY - panStartY);
    applyTransform();
    wrap.style.cursor = "grabbing";
  };
  State._ivcWindowUpHandler = () => {
    if (!isPanning) return;
    isPanning = false;
    const max = getMaxPan();
    wrap.style.cursor = max.x > 0 || max.y > 0 ? "grab" : "";
  };
  window.addEventListener("mousemove", State._ivcWindowMoveHandler);
  window.addEventListener("mouseup", State._ivcWindowUpHandler);
}

// ---- Waveform drawing --------------------------------------------------

export async function drawWaveform(mediaId) {
  const canvas = document.getElementById("waveform-canvas");
  if (!canvas) return;

  const ctx = canvas.getContext("2d");
  const rect = canvas.getBoundingClientRect();
  if (rect.width > 0) canvas.width = Math.round(rect.width);
  const width = canvas.width;
  const height = canvas.height;

  ctx.fillStyle = themeColor("--bg-surface");
  ctx.fillRect(0, 0, width, height);

  try {
    const response = await fetch(`/api/medias/${mediaId}/audio`);
    const arrayBuffer = await response.arrayBuffer();

    if (!State.waveformAudioCtx || State.waveformAudioCtx.state === "closed") {
      State.waveformAudioCtx = new (window.AudioContext || window.webkitAudioContext)();
    }
    const audioBuffer = await State.waveformAudioCtx.decodeAudioData(arrayBuffer);
    const channelData = audioBuffer.getChannelData(0);
    const step = Math.ceil(channelData.length / width);
    const amp = height / 2;

    ctx.strokeStyle = themeColor("--accent");
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let i = 0; i < width; i++) {
      let min = 1.0, max = -1.0;
      for (let j = 0; j < step; j++) {
        const datum = channelData[i * step + j];
        if (datum < min) min = datum;
        if (datum > max) max = datum;
      }
      const yMin = (1 + min) * amp;
      const yMax = (1 + max) * amp;
      if (i === 0) ctx.moveTo(i, yMin);
      ctx.lineTo(i, yMin);
      ctx.lineTo(i, yMax);
    }
    ctx.stroke();

    ctx.strokeStyle = themeColor("--border");
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, height / 2);
    ctx.lineTo(width, height / 2);
    ctx.stroke();
  } catch (error) {
    console.error("Error drawing waveform:", error);
    ctx.fillStyle = themeColor("--color-bad");
    ctx.font = "12px monospace";
    ctx.textAlign = "center";
    ctx.fillText("Unable to load waveform", width / 2, height / 2);
  }
}

export { mediaList, center };
