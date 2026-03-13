import { Component, EventEmitter, Input, OnInit, Output, OnChanges, SimpleChanges } from '@angular/core';
import { CommonModule } from '@angular/common';
import { MediaItem } from '../../../models/api.models';
import { LabelSortMode } from '../label-sort/label-sort.component';

export interface LabelEntry {
  id: number;
  name: string;
  time: number;
  score: number;
  confidence: number;
}

@Component({
  selector: 'vt-label-list',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './label-list.component.html',
  styleUrl: './label-list.component.scss',
})
export class LabelListComponent implements OnInit, OnChanges {
  @Input() label: 'good' | 'bad' = 'good';
  @Input() ids: number[] = [];
  @Input() medias: MediaItem[] = [];
  @Input() clickTimes: Record<string, number> = {};
  @Input() learnedScores: Record<string, number> = {};
  @Input() sortMode: LabelSortMode = 'time-desc';
  @Input() viewMode: 'grid' | 'list' = 'grid';
  @Input() gridColumns: number = 2;
  @Input() focusMode: 'click' | 'hover' = 'click';
  @Output() mediaSelected = new EventEmitter<number>();
  @Output() mediaVote = new EventEmitter<{ id: number; vote: 'good' | 'bad' }>();

  sortedEntries: LabelEntry[] = [];

  ngOnInit(): void {
    this.sortedEntries = this.buildSortedEntries();
  }

  ngOnChanges(_changes: SimpleChanges): void {
    this.sortedEntries = this.buildSortedEntries();
  }

  private buildSortedEntries(): LabelEntry[] {
    const entries = this.ids.map(id => this.buildEntry(id));
    return this.sortEntries(entries);
  }

  private buildEntry(id: number): LabelEntry {
    const media = this.medias.find(m => m.id === id);
    const name = media ? (media.filename || `Clip #${id}`) : `Clip #${id}`;
    const time = this.clickTimes[String(id)] ?? -1;
    const score = this.learnedScores[String(id)] ?? -1;
    let confidence = -1;
    if (score >= 0) {
      confidence = this.label === 'good' ? score : 1 - score;
    }
    return { id, name, time, score, confidence };
  }

  private sortEntries(entries: LabelEntry[]): LabelEntry[] {
    const sorted = [...entries];
    switch (this.sortMode) {
      case 'time-desc':
        sorted.sort((a, b) => b.time - a.time);
        break;
      case 'time-asc':
        sorted.sort((a, b) => a.time - b.time);
        break;
      case 'name-asc':
        sorted.sort((a, b) => a.name.localeCompare(b.name));
        break;
      case 'name-desc':
        sorted.sort((a, b) => b.name.localeCompare(a.name));
        break;
      case 'confidence-desc':
        sorted.sort((a, b) => b.confidence - a.confidence);
        break;
      case 'confidence-asc':
        sorted.sort((a, b) => a.confidence - b.confidence);
        break;
      case 'id-asc':
      default:
        sorted.sort((a, b) => a.id - b.id);
        break;
    }
    return sorted;
  }

  metaText(entry: LabelEntry): string {
    const parts: string[] = [];
    if (this.sortMode === 'time-desc' || this.sortMode === 'time-asc') {
      if (entry.time >= 0) {
        parts.push(`#${entry.time}`);
      } else {
        parts.push('imported');
      }
    } else if (this.sortMode === 'confidence-desc' || this.sortMode === 'confidence-asc') {
      if (entry.confidence >= 0) {
        parts.push(`${(entry.confidence * 100).toFixed(0)}%`);
      }
    }
    return parts.join(' \u00B7 ');
  }

  get isGrid(): boolean {
    return this.viewMode === 'grid';
  }

  hasThumbnailUrl(id: number): boolean {
    const media = this.medias.find(m => m.id === id);
    return !!media && (media.type === 'image' || media.type === 'video' || media.type === 'document');
  }

  isVideo(id: number): boolean {
    const media = this.medias.find(m => m.id === id);
    return !!media && media.type === 'video';
  }

  thumbnailUrl(id: number): string {
    const media = this.medias.find(m => m.id === id);
    if (!media) return '';
    if (media.type === 'video') return `/api/medias/${id}/video`;
    return `/api/medias/${id}/image`;
  }

  placeholderIcon(id: number): string | null {
    if (!this.isGrid) return null;
    const media = this.medias.find(m => m.id === id);
    if (!media) return null;
    if (media.type === 'image' || media.type === 'video' || media.type === 'document') return null;
    if (media.type === 'audio') return '\u266B';
    if (media.type === 'paragraph') return '\u00B6';
    return '\u25A1';
  }

  onEntryClick(id: number): void {
    if (this.focusMode === 'hover') {
      this.mediaVote.emit({ id, vote: 'bad' });
    } else {
      this.mediaSelected.emit(id);
    }
  }

  onEntryContextMenu(event: MouseEvent, id: number): void {
    if (this.focusMode === 'hover') {
      event.preventDefault();
      this.mediaVote.emit({ id, vote: 'good' });
    }
  }

  onEntryMouseEnter(id: number): void {
    if (this.focusMode === 'hover') {
      this.mediaSelected.emit(id);
    }
  }

  onEntryKeydown(event: KeyboardEvent, id: number): void {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      this.mediaSelected.emit(id);
    }
  }

  trackById(_index: number, entry: LabelEntry): number {
    return entry.id;
  }
}
