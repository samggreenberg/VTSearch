import { Injectable, signal } from '@angular/core';

export type DialogType = 'info' | 'warning' | 'error' | 'success';

interface DialogButton {
  label: string;
  primary: boolean;
  value: unknown;
}

/**
 * VtDialogService: Angular replacement for dialogs.js.
 * Provides confirm() and prompt() returning Promises. Pure
 * informational notifications go through ToastService instead of a
 * modal alert.
 *
 * A message may contain `\n`; the dialog host renders each one as a line
 * break. Use them to keep a question on its own line, apart from the list
 * it enumerates and the detail that explains it, rather than running all
 * three together into one sentence.
 */
@Injectable({ providedIn: 'root' })
export class VtDialogService {
  private activeResolve: ((value: unknown) => void) | null = null;

  // State for the current dialog, consumed by the dialog host component. These
  // are signals so a `show()` invoked from a non-event callback (e.g. a `.then()`
  // continuation, as in find-view's rename-after-prompt) still schedules change
  // detection and the dialog actually appears under zoneless. `dialogInputValue`
  // is written by the host's `[(ngModel)]` (via `.set()` in the template).
  readonly dialogOpen = signal(false);
  readonly dialogTitle = signal('');
  readonly dialogMessage = signal('');
  readonly dialogType = signal<DialogType>('info');
  readonly dialogShowInput = signal(false);
  readonly dialogInputValue = signal('');
  readonly dialogButtons = signal<DialogButton[]>([]);

  private static readonly ICON_TYPES: Record<DialogType, string> = {
    warning: 'warning',
    error: 'x-circle',
    success: 'check',
    info: 'info',
  };

  getIconType(): string {
    return VtDialogService.ICON_TYPES[this.dialogType()] || VtDialogService.ICON_TYPES.info;
  }

  confirm(message: string, type: DialogType = 'warning'): Promise<boolean> {
    return this.show({
      message,
      type,
      showInput: false,
      buttons: [
        { label: 'Cancel', primary: false, value: false },
        { label: 'OK', primary: true, value: true },
      ],
    }) as Promise<boolean>;
  }

  /**
   * Standardised confirmation for destructive actions.
   *
   * `question` should name the operation and its target, e.g.
   *   "Delete detector 'cats'?"
   * When it enumerates several targets, put the list on a line of its own:
   *   "Delete 2 detectors?\n'cats', 'dogs'"
   * `detail` should explain what is removed and what is unaffected, e.g.
   *   "(This deletes your labels. The underlying media is unaffected.)"
   * It is shown on its own line below the question.
   * `actionLabel` is the verb on the primary button (default "Delete").
   */
  confirmDestructive(question: string, detail: string, actionLabel = 'Delete'): Promise<boolean> {
    return this.show({
      message: VtDialogService.questionWithDetail(question, detail),
      type: 'warning',
      showInput: false,
      buttons: [
        { label: 'Cancel', primary: false, value: false },
        { label: actionLabel, primary: true, value: true },
      ],
    }) as Promise<boolean>;
  }

  /**
   * Like {@link confirmDestructive}, but inserts a middle "escape hatch"
   * button (e.g. "Export first…") between Cancel and the destructive action,
   * so the user can save their state before committing. Resolves:
   *   'confirm' — the destructive action was chosen
   *   'escape'  — the escape-hatch action was chosen
   *   'cancel'  — Cancel, Escape key, or backdrop dismissal
   * Cancel stays the first non-primary button so backdrop/Escape dismissal
   * (which resolves the first non-primary value) still reads as 'cancel'.
   */
  confirmDestructiveWithEscape(
    question: string,
    detail: string,
    actionLabel: string,
    escapeLabel: string,
  ): Promise<'confirm' | 'escape' | 'cancel'> {
    return this.show({
      message: VtDialogService.questionWithDetail(question, detail),
      type: 'warning',
      showInput: false,
      buttons: [
        { label: 'Cancel', primary: false, value: 'cancel' },
        { label: escapeLabel, primary: false, value: 'escape' },
        { label: actionLabel, primary: true, value: 'confirm' },
      ],
    }) as Promise<'confirm' | 'escape' | 'cancel'>;
  }

  prompt(message: string, defaultValue = '', type: DialogType = 'info'): Promise<string | null> {
    return this.show({
      message,
      type,
      showInput: true,
      inputDefault: defaultValue,
      buttons: [
        { label: 'Cancel', primary: false, value: null },
        { label: 'OK', primary: true, value: '__input__' },
      ],
    }) as Promise<string | null>;
  }

  /** The question on one line, the detail on the next. */
  private static questionWithDetail(question: string, detail: string): string {
    return `${question}\n${detail}`;
  }

  /** Resolve the current dialog with a value. Called by dialog host component. */
  resolve(value: unknown): void {
    if (this.activeResolve) {
      const resolvedValue = value === '__input__' ? this.dialogInputValue() : value;
      this.activeResolve(resolvedValue);
      this.activeResolve = null;
      this.dialogOpen.set(false);
    }
  }

  /**
   * Dismiss the current dialog as if its non-primary (Cancel) button was
   * clicked. Escape / backdrop dismissal must use the dialog kind's own
   * cancel value: a `prompt()` resolves `null` (its callers are typed
   * `string | null`, and a hard-coded `false` made `(name ?? '').trim()` /
   * `result.split(...)` throw), while `confirm()` resolves `false`.
   */
  cancel(): void {
    const cancelButton = this.dialogButtons().find((btn) => !btn.primary);
    this.resolve(cancelButton ? cancelButton.value : false);
  }

  private show(config: {
    message: string;
    type: DialogType;
    showInput: boolean;
    inputDefault?: string;
    buttons: DialogButton[];
  }): Promise<unknown> {
    // A second show() while a dialog is still pending used to overwrite
    // `activeResolve`, stranding the first caller's promise forever. Settle
    // the superseded dialog as cancelled instead.
    if (this.activeResolve) {
      this.cancel();
    }
    return new Promise(resolve => {
      this.activeResolve = resolve;
      this.dialogTitle.set('');
      this.dialogMessage.set(config.message);
      this.dialogType.set(config.type);
      this.dialogShowInput.set(config.showInput);
      this.dialogInputValue.set(config.inputDefault || '');
      this.dialogButtons.set(config.buttons);
      this.dialogOpen.set(true);
    });
  }
}
