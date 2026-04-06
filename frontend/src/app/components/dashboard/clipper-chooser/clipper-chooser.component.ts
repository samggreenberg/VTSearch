import { Component, EventEmitter, Input, OnChanges, Output, SimpleChanges } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ModalComponent } from '../../modal/modal.component';
import { ClipperInfo, ClipperParameter } from '../../../models/api.models';

export interface ClipperSelection {
  name: string;
  params: Record<string, number | string>;
}

@Component({
  selector: 'vt-clipper-chooser',
  standalone: true,
  imports: [CommonModule, FormsModule, ModalComponent],
  templateUrl: './clipper-chooser.component.html',
  styleUrl: './clipper-chooser.component.scss',
})
export class ClipperChooserComponent implements OnChanges {
  @Input() open = false;
  @Input() clippers: ClipperInfo[] = [];

  @Output() selected = new EventEmitter<ClipperSelection>();
  @Output() cancelled = new EventEmitter<void>();

  activeTab = '';
  /** Per-clipper parameter values, keyed by clipper name then param key. */
  paramValues: Record<string, Record<string, number | string>> = {};

  ngOnChanges(changes: SimpleChanges): void {
    if (changes['open'] && this.open) {
      this.initTabs();
    }
  }

  private initTabs(): void {
    this.paramValues = {};
    for (const clipper of this.clippers) {
      const questions = clipper.creation_questions || clipper.parameters || [];
      const vals: Record<string, number | string> = {};
      for (const q of questions) {
        vals[q.key] = q.default;
      }
      this.paramValues[clipper.name] = vals;
    }
    // Default to first non-default clipper tab if available, else first clipper
    if (this.clippers.length > 0) {
      const nonDefault = this.clippers.find((c) => !c.name.endsWith('_default'));
      this.activeTab = nonDefault ? nonDefault.name : this.clippers[0].name;
    }
  }

  get activeClipper(): ClipperInfo | undefined {
    return this.clippers.find((c) => c.name === this.activeTab);
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
