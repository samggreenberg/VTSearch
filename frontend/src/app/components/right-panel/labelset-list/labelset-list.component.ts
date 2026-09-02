import { ChangeDetectionStrategy, Component, computed, inject, input, output } from '@angular/core';
import { CommonModule } from '@angular/common';
import type { DetectorLabelView } from '../../../generated/api-client/models/detector-label-view';
import { LabelSortMode } from '../label-sort/label-sort.component';
import { DetectorsCrudApiService } from '../../../services/detectors-crud-api.service';
import { MediaTypeCapabilityService } from '../../../services/media-type-capability.service';
import { VoteGridComponent, VoteGridEntry } from '../vote-grid/vote-grid.component';
import { sortListEntries } from '../../../utils/sort-list-entries';

interface SortedElement extends DetectorLabelView, VoteGridEntry {
  confidence: number;
}

@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-labelset-list',
  standalone: true,
  imports: [CommonModule, VoteGridComponent],
  templateUrl: './labelset-list.component.html',
  styleUrls: ['../label-list/label-list.component.scss'],
})
export class LabelsetListComponent {
  private detectorsCrudApi = inject(DetectorsCrudApiService);
  private mediaTypeCaps = inject(MediaTypeCapabilityService);

  readonly label = input<'good' | 'bad'>('good');
  readonly elements = input<DetectorLabelView[]>([]);
  readonly modelName = input<string>('');
  readonly sortMode = input<LabelSortMode>('time-desc');
  readonly gridGoalWidth = input<number>(80);
  readonly focusMode = input<'click' | 'hover'>('click');
  readonly elementSelected = output<DetectorLabelView>();
  readonly elementVote = output<{
    id: string;
    vote: 'good' | 'bad';
}>();

  // Recomputes when any of the inputs it reads change (elements, label,
  // modelName, sortMode); signal inputs don't fire ngOnChanges.
  readonly sortedEntries = computed<SortedElement[]>(() => this.buildSorted());

  private buildSorted(): SortedElement[] {
    const entries: SortedElement[] = this.elements().map((e) => ({
      ...e,
      key: e.id,
      confidence: e.score >= 0 ? (this.label() === 'good' ? e.score : 1 - e.score) : -1,
      thumbnailUrl: this.buildThumbnailUrl(e),
      fallbackIcon: e.media_type === 'audio' ? '♫' : e.media_type === 'text' ? '¶' : '□',
      missing: e.cid === null || e.cid === undefined,
      // Audio waveforms are theme-agnostic alpha masks (issue #2369); flag them
      // so vote-grid tints them via a CSS mask instead of a plain <img>.
      isAudio: e.media_type === 'audio',
    }));
    return sortListEntries(entries, this.sortMode());
  }

  private buildThumbnailUrl(entry: DetectorLabelView): string {
    const modelName = this.modelName();
    if (!modelName || !this.mediaTypeCaps.usesThumbnails(entry.media_type)) return '';
    let url = this.detectorsCrudApi.labelThumbnailUrl(modelName, entry.id);
    // The route crops to the element's stored region box server-side; fold the
    // box into the URL so a re-vote with a different box busts the cached tile
    // (the box coords aren't otherwise part of the URL).
    const box = entry.region_box;
    if (box && box.length === 4) {
      const sep = url.includes('?') ? '&' : '?';
      url += `${sep}region=${box.map((v) => v.toFixed(4)).join(',')}`;
    }
    return url;
  }

  onEntrySelected(entry: VoteGridEntry): void {
    this.elementSelected.emit(entry as SortedElement);
  }

  onEntryVote(event: { entry: VoteGridEntry; vote: 'good' | 'bad' }): void {
    this.elementVote.emit({ id: event.entry.key, vote: event.vote });
  }
}
