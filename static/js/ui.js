/**
 * UI utilities: escapeHtml, screen-reader announcer, theme helpers,
 * ETA formatting, and the custom VTSearch dialog system.
 */

// ---- HTML escaping (XSS prevention) ------------------------------------

export function escapeHtml(text) {
  if (text == null) return "";
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML.replace(/'/g, "&#39;").replace(/"/g, "&quot;");
}

// ---- Screen-reader announcer -------------------------------------------

const srAnnouncer = document.createElement("div");
srAnnouncer.setAttribute("aria-live", "polite");
srAnnouncer.setAttribute("aria-atomic", "true");
srAnnouncer.className = "sr-only";
srAnnouncer.id = "sr-announcer";
document.body.appendChild(srAnnouncer);

export function announce(message) {
  srAnnouncer.textContent = "";
  setTimeout(() => { srAnnouncer.textContent = message; }, 100);
}

// ---- Theme helpers -----------------------------------------------------

export function themeColor(varName) {
  return getComputedStyle(document.documentElement).getPropertyValue(varName).trim();
}

const themeBtns = document.querySelectorAll(".theme-btn");

export function applyTheme(theme) {
  if (theme === "light" || theme === "highviz") {
    document.documentElement.setAttribute("data-theme", theme);
  } else {
    document.documentElement.removeAttribute("data-theme");
  }
  themeBtns.forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.theme === theme);
  });
}

// ---- ETA formatting ----------------------------------------------------

export function formatETA(secondsRemaining) {
  if (secondsRemaining < 5) return "Less than 5 seconds remaining";
  if (secondsRemaining < 30) return "Less than 30 seconds remaining";
  if (secondsRemaining < 60) return "Less than a minute remaining";
  if (secondsRemaining < 90) return "About a minute remaining";
  if (secondsRemaining < 3600)
    return `About ${Math.round(secondsRemaining / 60)} minutes remaining`;
  const hours = Math.floor(secondsRemaining / 3600);
  const mins = Math.round((secondsRemaining % 3600) / 60);
  if (hours === 1)
    return mins > 0 ? `About 1 hour ${mins} minutes remaining` : "About 1 hour remaining";
  return `About ${hours} hours remaining`;
}

// ---- Custom VTSearch dialog system -------------------------------------

const vtDialogModal   = document.getElementById("vt-dialog-modal");
const vtDialogIcon    = document.getElementById("vt-dialog-icon");
const vtDialogMessage = document.getElementById("vt-dialog-message");
const vtDialogInput   = document.getElementById("vt-dialog-input");
const vtDialogActions = document.getElementById("vt-dialog-actions");

const VT_ICONS = {
  warning: "\u26A0\uFE0F",
  error:   "\u274C",
  success: "\u2705",
  info:    "\u2139\uFE0F",
};

export function vtShowDialog({ message, type, showInput, inputDefault, buttons }) {
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

export function vtAlert(message, type) {
  type = type || "info";
  return vtShowDialog({
    message,
    type,
    showInput: false,
    buttons: [{ label: "OK", primary: true, value: true }],
  });
}

export function vtConfirm(message, type) {
  type = type || "warning";
  return vtShowDialog({
    message,
    type,
    showInput: false,
    buttons: [
      { label: "Cancel", primary: false, value: false },
      { label: "OK",     primary: true,  value: true },
    ],
  });
}

export function vtPrompt(message, defaultValue, type) {
  type = type || "info";
  return vtShowDialog({
    message,
    type,
    showInput: true,
    inputDefault: defaultValue || "",
    buttons: [
      { label: "Cancel", primary: false, value: null },
      { label: "OK",     primary: true,  value: "input" },
    ],
  });
}

export function isDialogOpen() {
  return vtDialogModal && vtDialogModal.classList.contains("show");
}

// ---- Typing guard ------------------------------------------------------

export function isTyping() {
  const el = document.activeElement;
  if (!el) return false;
  const tag = el.tagName;
  if (tag === "INPUT" && el.type !== "checkbox" && el.type !== "radio" && el.type !== "range") return true;
  if (tag === "TEXTAREA" || tag === "SELECT") return true;
  if (el.isContentEditable) return true;
  return false;
}

// ---- Download helper ---------------------------------------------------

export function triggerDownload(content, filename, mimeType) {
  const blob = new Blob([content], { type: mimeType || "application/octet-stream" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}
