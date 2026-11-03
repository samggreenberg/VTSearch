import { ChangeDetectionStrategy, Component, computed, input, output } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ModalComponent } from '../../modal/modal.component';
import { ClipboardColumn, ClipboardCopyComponent } from '../../clipboard-copy/clipboard-copy.component';
import type { DroppedItemsGroup } from '../../../generated/api-client/models/dropped-items-group';

/** One table/clipboard row: a dropped item and why it was dropped. */
export interface DroppedItemRow {
  item: string;
  reason: string;
}

/**
 * The items a dataset lost for want of a vector (issue #3798): the ones its
 * import could not embed, and any a later load removed because they were
 * missing a vector the rest of the dataset has. The import's toast names them
 * once; this lists them for as long as the dataset exists. Opened as a child
 * modal from the Dataset Stats modal's "Dropped items" row, which already holds
 * the groups, so there is nothing to fetch.
 */
@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-dropped-items-modal',
  standalone: true,
  imports: [CommonModule, ModalComponent, ClipboardCopyComponent],
  templateUrl: './dropped-items-modal.component.html',
  styleUrl: './dropped-items-modal.component.scss',
})
export class DroppedItemsModalComponent {
  readonly datasetName = input('');
  readonly groups = input<DroppedItemsGroup[]>([]);
  readonly closed = output<void>();

  readonly clipboardColumns: ClipboardColumn[] = [
    { key: 'item', label: 'Item' },
    { key: 'reason', label: 'Reason' },
  ];

  /** Every recorded item, one row each, commonest reason first. */
  readonly rows = computed<DroppedItemRow[]>(() =>
    this.groups().flatMap((g) => g.items.map((item) => ({ item, reason: g.reason }))),
  );

  readonly total = computed(() => this.groups().reduce((n, g) => n + g.count, 0));

  /** Dropped items past the server's name cap: counted, but not named. */
  readonly unlisted = computed(() => this.total() - this.rows().length);

  get clipboardRows(): Record<string, string>[] {
    return this.rows() as unknown as Record<string, string>[];
  }

  close(): void {
    this.closed.emit();
  }
}
