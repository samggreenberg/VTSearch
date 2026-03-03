/**
 * VTSearch dialog system — custom modal dialogs replacing native
 * alert/confirm/prompt with themed, accessible alternatives.
 *
 * Exposed on window.VTDialogs so other modules can call them.
 */
(function () {
  "use strict";

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

  window.VTDialogs = {
    vtShowDialog,
    vtAlert,
    vtConfirm,
    vtPrompt,
  };
})();
