/**
 * Generic PluginModal — replaces the 4 nearly-identical modal workflows for
 * label importers, processor importers, dataset importers, and label/detector
 * exporters.
 *
 * Each workflow follows the same pattern:
 *   1. Fetch a list of plugins from an API endpoint
 *   2. Render them as option cards in a modal
 *   3. On click, show a form with the plugin's fields
 *   4. On submit, POST a FormData or JSON body to an import endpoint
 *   5. Show status, close modal on success
 *
 * PluginModal unifies all of this behind a single configurable class.
 */

import { escapeHtml } from "./ui.js";

/**
 * Build the inner HTML for a set of option cards.
 *
 * @param {Array} plugins      - [{name, icon, display_name, description, ...}]
 * @param {string} optionClass - CSS class for each card (e.g. "label-importer-option")
 * @returns {string} HTML
 */
export function renderOptionCards(plugins, optionClass) {
  return plugins
    .map(
      (p) => `
    <div class="${optionClass} option-card" data-name="${escapeHtml(p.name)}" role="button" tabindex="0">
      <span class="option-card-icon">${escapeHtml(p.icon || "\uD83D\uDD0C")}</span>
      <div>
        <div class="option-card-title">${escapeHtml(p.display_name)}</div>
        <div class="option-card-desc">${escapeHtml(p.description)}</div>
      </div>
    </div>`,
    )
    .join("");
}

/**
 * Build form HTML from a plugin's field descriptors.
 *
 * @param {object}  plugin          - The plugin metadata with .fields array
 * @param {object}  opts
 * @param {string}  opts.formId     - <form> id attribute
 * @param {string}  opts.statusId   - Status text element id
 * @param {string}  opts.submitLabel - Submit button text (default "Import")
 * @param {boolean} opts.showNameField - Prepend a "Name *" text input
 * @param {string}  opts.nameLabel  - Label for the name field (default "Detector Name *")
 * @param {string}  opts.namePlaceholder - placeholder (default "e.g. My Detector")
 * @param {string}  opts.nameHint   - hint text
 * @param {string}  opts.headingPrefix - text before display_name in <h3>
 * @returns {string} HTML
 */
export function renderPluginForm(plugin, opts = {}) {
  const {
    formId = "plugin-form",
    statusId = "plugin-status",
    submitLabel = "Import",
    showNameField = false,
    nameLabel = "Detector Name *",
    namePlaceholder = "e.g. My Detector",
    nameHint = "Name for the imported detector.",
    headingPrefix = "",
  } = opts;

  let html = `<h3 class="form-heading">${headingPrefix}${escapeHtml(plugin.display_name)}</h3>`;
  html += `<form id="${formId}">`;

  if (showNameField) {
    html += `<div class="form-group">`;
    html += `<label class="form-label">${escapeHtml(nameLabel)}</label>`;
    html += `<input type="text" name="name" placeholder="${escapeHtml(namePlaceholder)}" class="form-input" required>`;
    if (nameHint) html += `<div class="form-hint">${escapeHtml(nameHint)}</div>`;
    html += `</div>`;
  }

  for (const field of plugin.fields) {
    html += `<div class="form-group">`;
    html += `<label class="form-label">${escapeHtml(field.label)}${field.required ? " *" : ""}</label>`;

    if (field.field_type === "file") {
      html += `<input type="file" name="${escapeHtml(field.key)}" accept="${escapeHtml(field.accept || "")}" class="form-input" ${field.required ? "required" : ""}>`;
    } else if (field.field_type === "select") {
      html += `<select name="${escapeHtml(field.key)}" class="form-input">`;
      for (const opt of field.options) {
        html += `<option value="${escapeHtml(opt)}"${opt === field.default ? " selected" : ""}>${escapeHtml(opt || "(auto-detect)")}</option>`;
      }
      html += `</select>`;
    } else if (field.field_type === "folder") {
      html += `<div class="form-row"><input type="text" name="${escapeHtml(field.key)}" placeholder="${escapeHtml(field.description)}" class="form-input" style="flex:1;" data-folder-input="true" ${field.required ? "required" : ""}>`;
      html += `<button type="button" data-browse-btn="true" class="btn-browse">Browse\u2026</button></div>`;
      html += `<input type="file" data-folder-picker="true" webkitdirectory style="display:none;">`;
    } else {
      const itype = field.field_type === "password" ? "password" : field.field_type === "url" ? "url" : "text";
      const placeholder = escapeHtml(field.placeholder || field.description);
      html += `<input type="${itype}" name="${escapeHtml(field.key)}" value="${escapeHtml(field.default || "")}" placeholder="${placeholder}" class="form-input" ${field.required ? "required" : ""}>`;
    }

    if (field.description) {
      html += `<div class="form-hint">${escapeHtml(field.description)}</div>`;
    }
    html += `</div>`;
  }

  html += `<div id="${statusId}" class="status-text compact"></div>`;
  html += `<button type="submit" class="btn-block-primary">${escapeHtml(submitLabel)}</button>`;
  html += `</form>`;
  return html;
}

/**
 * Wire up folder-browse button pairs inside a container element.
 */
export function wireFolderBrowse(container) {
  const browseBtn = container.querySelector("[data-browse-btn]");
  const folderPicker = container.querySelector("[data-folder-picker]");
  const folderTextInput = container.querySelector("[data-folder-input]");
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
}

/**
 * Collect form data for a plugin submission.
 * Returns { body, headers } ready for fetch().
 *
 * @param {HTMLFormElement} formEl
 * @param {object} plugin           - Plugin metadata (to inspect field_types)
 * @param {string[]} extraJsonKeys  - Additional keys to include in JSON body (e.g. ["name"])
 */
export function collectFormData(formEl, plugin, extraJsonKeys = []) {
  const hasFiles = plugin.fields.some((f) => f.field_type === "file");
  if (hasFiles) {
    return { body: new FormData(formEl), headers: {} };
  }
  const obj = {};
  for (const key of extraJsonKeys) {
    if (formEl.elements[key]) obj[key] = formEl.elements[key].value;
  }
  for (const field of plugin.fields) {
    obj[field.key] = formEl.elements[field.key].value;
  }
  return {
    body: JSON.stringify(obj),
    headers: { "Content-Type": "application/json" },
  };
}

/**
 * PluginModal — manages the open/back/close lifecycle and list→form transition
 * for any plugin-picker modal.
 */
export class PluginModal {
  /**
   * @param {object} els           DOM element references:
   * @param {Element} els.modal    The .modal container
   * @param {Element} els.close    The close button
   * @param {Element} els.list     The plugin-list container
   * @param {Element} els.form     The form container
   * @param {Element} els.back     The back button
   */
  constructor(els) {
    this.modal = els.modal;
    this.listEl = els.list;
    this.formEl = els.form;
    this.backEl = els.back;

    if (els.close) {
      els.close.addEventListener("click", () => this.close());
    }
    if (els.back) {
      els.back.addEventListener("click", () => this.showList());
    }
  }

  open() {
    this.modal.classList.add("show");
  }

  close() {
    this.modal.classList.remove("show");
  }

  /** Reset to the list view. */
  showList() {
    if (this.formEl) {
      this.formEl.style.display = "none";
      this.formEl.innerHTML = "";
    }
    if (this.backEl) this.backEl.style.display = "none";
    if (this.listEl) this.listEl.style.display = "";
  }

  /** Transition from list → form. */
  showForm(html) {
    if (this.listEl) this.listEl.style.display = "none";
    if (this.backEl) this.backEl.style.display = "inline-block";
    if (this.formEl) {
      this.formEl.innerHTML = html;
      this.formEl.style.display = "block";
      wireFolderBrowse(this.formEl);
    }
  }

  /**
   * Populate the list with option cards, wire click→onSelect callbacks.
   *
   * @param {Array}    plugins      Plugin metadata array
   * @param {string}   optionClass  CSS class for cards
   * @param {Function} onSelect     Called with (plugin) when a card is clicked
   * @param {string}   [extraHtml]  HTML appended after the option cards
   */
  populateList(plugins, optionClass, onSelect, extraHtml = "") {
    this.showList();
    if (plugins.length === 0 && !extraHtml) {
      this.listEl.innerHTML = '<p style="color:var(--text-muted);">None available.</p>';
      return;
    }

    this.listEl.innerHTML = renderOptionCards(plugins, optionClass) + extraHtml;

    this.listEl.querySelectorAll(`.${optionClass}`).forEach((el) => {
      el.addEventListener("click", () => {
        const p = plugins.find((x) => x.name === el.dataset.name);
        if (p) onSelect(p);
      });
      el.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          el.click();
        }
      });
    });
  }
}
