import { Injectable, ComponentRef, ApplicationRef, createComponent, EnvironmentInjector } from '@angular/core';
import { ModalComponent } from '../components/modal/modal.component';

export type DialogType = 'info' | 'warning' | 'error' | 'success';

interface DialogButton {
  label: string;
  primary: boolean;
  value: unknown;
}

/**
 * VtDialogService — Angular replacement for dialogs.js.
 * Provides confirm() and prompt() returning Promises. Pure
 * informational notifications go through ToastService instead of a
 * modal alert.
 */
@Injectable({ providedIn: 'root' })
export class VtDialogService {
  private activeResolve: ((value: unknown) => void) | null = null;
  private modalRef: ComponentRef<ModalComponent> | null = null;

  // State for the current dialog — consumed by a dialog host component.
  dialogOpen = false;
  dialogTitle = '';
  dialogMessage = '';
  dialogType: DialogType = 'info';
  dialogShowInput = false;
  dialogInputValue = '';
  dialogButtons: DialogButton[] = [];

  private static readonly ICON_TYPES: Record<DialogType, string> = {
    warning: 'warning',
    error: 'x-circle',
    success: 'check',
    info: 'info',
  };

  getIconType(): string {
    return VtDialogService.ICON_TYPES[this.dialogType] || VtDialogService.ICON_TYPES.info;
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
   * `detail` should explain what is removed and what is unaffected, e.g.
   *   "(This deletes your labels. The underlying media is unaffected.)"
   * `actionLabel` is the verb on the primary button (default "Delete").
   */
  confirmDestructive(question: string, detail: string, actionLabel = 'Delete'): Promise<boolean> {
    return this.show({
      message: `${question} ${detail}`,
      type: 'warning',
      showInput: false,
      buttons: [
        { label: 'Cancel', primary: false, value: false },
        { label: actionLabel, primary: true, value: true },
      ],
    }) as Promise<boolean>;
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

  /** Resolve the current dialog with a value. Called by dialog host component. */
  resolve(value: unknown): void {
    if (this.activeResolve) {
      const resolvedValue = value === '__input__' ? this.dialogInputValue : value;
      this.activeResolve(resolvedValue);
      this.activeResolve = null;
      this.dialogOpen = false;
    }
  }

  private show(config: {
    message: string;
    type: DialogType;
    showInput: boolean;
    inputDefault?: string;
    buttons: DialogButton[];
  }): Promise<unknown> {
    return new Promise(resolve => {
      this.activeResolve = resolve;
      this.dialogTitle = '';
      this.dialogMessage = config.message;
      this.dialogType = config.type;
      this.dialogShowInput = config.showInput;
      this.dialogInputValue = config.inputDefault || '';
      this.dialogButtons = config.buttons;
      this.dialogOpen = true;
    });
  }
}
