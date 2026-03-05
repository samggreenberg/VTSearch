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
 * Provides alert(), confirm(), and prompt() returning Promises.
 *
 * Uses a lightweight approach: creates temporary ModalComponent instances.
 * For Phase 2 this keeps things simple; a more elaborate approach using
 * dynamic components can be added later if needed.
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

  private static readonly ICONS: Record<DialogType, string> = {
    warning: '\u26A0\uFE0F',
    error: '\u274C',
    success: '\u2705',
    info: '\u2139\uFE0F',
  };

  getIcon(): string {
    return VtDialogService.ICONS[this.dialogType] || VtDialogService.ICONS.info;
  }

  alert(message: string, type: DialogType = 'info'): Promise<boolean> {
    return this.show({
      message,
      type,
      showInput: false,
      buttons: [{ label: 'OK', primary: true, value: true }],
    }) as Promise<boolean>;
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
