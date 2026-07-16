import { ChangeDetectionStrategy, Component, effect, input, output } from '@angular/core';

import { FormsModule } from '@angular/forms';
import { ModalComponent } from '../../modal/modal.component';
import { ClipperInfo, ClipperParameter } from '../../../models/api.models';

export interface ClipperSelection {
  name: string;
  params: Record<string, number | string>;
}

@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-clipper-chooser',
  standalone: true,
  imports: [FormsModule, ModalComponent],
  templateUrl: './clipper-chooser.component.html',
  styleUrl: './clipper-chooser.component.scss',
})
export class ClipperChooserComponent {
  readonly open = input(false);
  readonly clippers = input<ClipperInfo[]>([]);

  readonly selected = output<ClipperSelection>();
  readonly cancelled = output<void>();

  activeTab = '';
  /** Per-clipper parameter values, keyed by clipper name then param key. */
  paramValues: Record<string, Record<string, number | string>> = {};

  private wasOpen = false;

  constructor() {
    effect(() => {
      const isOpen = this.open();
      if (isOpen && !this.wasOpen) {
        this.initTabs();
      }
      this.wasOpen = isOpen;
    });
  }

  private initTabs(): void {
    this.paramValues = {};
    for (const clipper of this.clippers()) {
      const questions = clipper.creation_questions || clipper.parameters || [];
      const vals: Record<string, number | string> = {};
      for (const q of questions) {
        vals[q.key] = q.default;
      }
      this.paramValues[clipper.name] = vals;
    }
    // Default to first non-default clipper tab if available, else first clipper
    if (this.clippers().length > 0) {
      const nonDefault = this.clippers().find((c) => !c.name.endsWith('_default'));
      this.activeTab = nonDefault ? nonDefault.name : this.clippers()[0].name;
    }
  }

  get activeClipper(): ClipperInfo | undefined {
    return this.clippers().find((c) => c.name === this.activeTab);
  }

  get activeQuestions(): ClipperParameter[] {
    const clipper = this.activeClipper;
    return clipper?.creation_questions || clipper?.parameters || [];
  }

  selectTab(clipperName: string): void {
    this.activeTab = clipperName;
  }

  confirm(): void {
    const params = this.paramValues[this.activeTab] || {};
    this.selected.emit({ name: this.activeTab, params: { ...params } });
  }

  cancel(): void {
    this.cancelled.emit();
  }
}
