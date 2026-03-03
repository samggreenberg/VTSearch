/**
 * VTSearch results modal — autodetect/find results display, copy-to-clipboard,
 * and exporter controls for the results modal.
 *
 * Exposed on window.VTResults so the main IIFE in app.js can call them.
 */
(function () {
  "use strict";

  // ---- Shared utilities ----

  function escapeHtml(text) {
    if (text == null) return "";
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML.replace(/'/g, '&#39;').replace(/"/g, '&quot;');
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

  // ---- DOM element references (captured on init) ----

  let autodetectSummary = null;
  let autodetectResults = null;
  let autodetectModal = null;
  let autodetectModalClose = null;
  let copyResultsBtn = null;
  let exportExporterSelect = null;
  let exportExporterFields = null;
  let exportRunBtn = null;
  let exportStatus = null;
  let fillFromSortCheckbox = null;
  let fillFromSortInfo = null;

  // ---- Module state ----

  let exportersList = [];

  // ---- Dependency callbacks (injected on init) ----

  let _deps = {};

  // ---- Export status helper ----

  function setExportStatus(msg, color) {
    if (exportStatus) {
      exportStatus.textContent = msg;
      exportStatus.style.color = color || "var(--text-muted)";
    }
  }

  // ---- Export side selection ----

  function getSelectedExportSides() {
    const checked = document.querySelector('input[name="export-sides"]:checked');
    return checked ? checked.value : "good";
  }

  // ---- Exporter list / dropdown / fields ----

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

  // ---- Filtered results builder ----

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

  // ---- Fill-from-sort info ----

  function updateFillFromSortInfo() {
    if (!fillFromSortInfo) return;
    if (!fillFromSortCheckbox || !fillFromSortCheckbox.checked) {
      fillFromSortInfo.textContent = "";
      return;
    }
    const sortOrder = _deps.getSortOrder();
    const threshold = _deps.getThreshold();
    const votes = _deps.getVotes();
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

  // ---- Autodetect summary update ----

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
        <strong>Media Type:</strong> ${escapeHtml(data.media_type)} &nbsp;|&nbsp;
        <strong>Detectors Run:</strong> ${data.detectors_run} &nbsp;|&nbsp;
        ${countText}
      </p>
    `;
  }

  // ---- Run exporter with results ----

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

  // ---- Display results ----

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
        <strong>Media Type:</strong> ${escapeHtml(data.media_type)} &nbsp;|&nbsp;
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

  function displayFindResults(data) {
    // Display multi-dataset, multi-model Find results
    const results = data.results || [];
    const modelNames = data.models || [];
    const datasetNames = data.datasets || [];
    const multiDs = data.multiple_datasets;
    const multiMs = data.multiple_models;

    // Summary
    autodetectSummary.innerHTML = `
      <p style="color: var(--text-primary);">
        <strong>Datasets:</strong> ${datasetNames.map(n => escapeHtml(n)).join(", ")} &nbsp;|&nbsp;
        <strong>Models:</strong> ${modelNames.map(n => escapeHtml(n)).join(", ")} &nbsp;|&nbsp;
        <strong>Good Results:</strong> ${results.length}
      </p>
    `;

    if (results.length === 0) {
      autodetectResults.innerHTML = '<p style="color: var(--text-muted);">No positive hits found.</p>';
    } else {
      let tableHtml = `<table class="results-table"><thead><tr>`;
      if (multiDs) tableHtml += `<th>Dataset</th>`;
      tableHtml += `<th>Name</th>`;
      tableHtml += `<th>MD5</th>`;
      if (multiMs) {
        for (const mn of modelNames) {
          tableHtml += `<th>${escapeHtml(mn)}</th>`;
        }
      }
      tableHtml += `</tr></thead><tbody>`;
      for (const hit of results) {
        const name = escapeHtml(hit.origin_name || hit.filename || "");
        const md5 = escapeHtml(hit.md5 || "");
        tableHtml += `<tr>`;
        if (multiDs) tableHtml += `<td class="col-secondary">${escapeHtml(hit.dataset_name || "")}</td>`;
        tableHtml += `<td>${name}</td>`;
        tableHtml += `<td class="col-muted">${md5}</td>`;
        if (multiMs) {
          for (const mn of modelNames) {
            const v = (hit.model_verdicts || {})[mn];
            if (v) {
              const cls = v.verdict === "Good" ? "style=\"color:var(--color-good)\"" : "";
              tableHtml += `<td ${cls}>${escapeHtml(v.verdict)}</td>`;
            } else {
              tableHtml += `<td>-</td>`;
            }
          }
        }
        tableHtml += `</tr>`;
      }
      tableHtml += `</tbody></table>`;
      autodetectResults.innerHTML = tableHtml;
    }

    // Store for copying
    window.autodetectResultsData = data;
    window.autodetectAllHits = results;

    // Reset export controls
    const goodRadio = document.querySelector('input[name="export-sides"][value="good"]');
    if (goodRadio) goodRadio.checked = true;
    if (fillFromSortCheckbox) fillFromSortCheckbox.checked = false;
    if (fillFromSortInfo) fillFromSortInfo.textContent = "";
    setExportStatus("", "var(--text-muted)");
    if (exportersList.length === 0) loadExportersList();
    autodetectModal.classList.add("show");
  }

  // ---- Initialization: capture DOM elements and wire event listeners ----

  function init(deps) {
    _deps = deps;

    // Capture DOM elements
    autodetectSummary = document.getElementById("autodetect-summary");
    autodetectResults = document.getElementById("autodetect-results");
    autodetectModal = document.getElementById("autodetect-modal");
    autodetectModalClose = document.getElementById("autodetect-modal-close");
    copyResultsBtn = document.getElementById("copy-results-btn");
    exportExporterSelect = document.getElementById("export-exporter-select");
    exportExporterFields = document.getElementById("export-exporter-fields");
    exportRunBtn = document.getElementById("export-run-btn");
    exportStatus = document.getElementById("export-status");
    fillFromSortCheckbox = document.getElementById("fill-from-sort-checkbox");
    fillFromSortInfo = document.getElementById("fill-from-sort-info");

    // Wire modal close
    if (autodetectModalClose) {
      autodetectModalClose.addEventListener("click", () => {
        autodetectModal.classList.remove("show");
      });
    }

    // Wire copy button
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

    // Wire exporter dropdown change
    if (exportExporterSelect) {
      exportExporterSelect.addEventListener("change", renderExporterFields);
    }

    // Wire fill-from-sort checkbox
    if (fillFromSortCheckbox) {
      fillFromSortCheckbox.addEventListener("change", updateFillFromSortInfo);
    }

    // Wire export-sides radio buttons
    document.querySelectorAll('input[name="export-sides"]').forEach(radio => {
      radio.addEventListener("change", () => {
        updateFillFromSortInfo();
        updateAutodetectSummary();
      });
    });

    // Wire export run button
    if (exportRunBtn) {
      exportRunBtn.addEventListener("click", async () => {
        const sides = getSelectedExportSides();
        const useFill = fillFromSortCheckbox && fillFromSortCheckbox.checked;

        if (useFill) {
          // Fill from Sort mode
          const sortOrder = _deps.getSortOrder();
          const threshold = _deps.getThreshold();
          if (!sortOrder || threshold === null) {
            await _deps.vtAlert("No sort results available. Run a sort first.", "warning");
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
            await _deps.vtAlert("No unlabeled elements to fill. All elements in the sort results are already labeled.", "info");
            return;
          }

          let desc;
          if (sides === "good") desc = `${counts.good_count} Good label${counts.good_count !== 1 ? "s" : ""}`;
          else if (sides === "bad") desc = `${counts.bad_count} Bad label${counts.bad_count !== 1 ? "s" : ""}`;
          else desc = `${counts.good_count} Good + ${counts.bad_count} Bad label${total !== 1 ? "s" : ""}`;

          const confirmed = await _deps.vtConfirm(`This will add ${desc} to the LabelSet and export. Continue?`);
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
              _deps.setVotes(vData);
              _deps.renderVotes();
            }
          } catch (_) {}
        } else {
          // Standard auto-detect export
          const filteredResults = buildFilteredResults(sides);
          await runExporterWithResults(filteredResults);
        }
      });
    }
  }

  // ---- Public API ----

  window.VTResults = {
    init,
    escapeHtml,
    formatOrigin,
    displayAutodetectResults,
    displayFindResults,
    // Expose for the Escape-key modal-close list in app.js
    getAutodetectModal: () => autodetectModal,
    getAutodetectModalClose: () => autodetectModalClose,
  };
})();
