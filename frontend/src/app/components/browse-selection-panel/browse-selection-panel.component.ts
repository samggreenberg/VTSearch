import { Component, OnDestroy, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Subscription } from 'rxjs';
import { BrowseSelectionService } from '../../services/browse-selection.service';

/**
 * Floating chip over the browse canvas that reports how many items are
 * currently selected and offers a one-click Clear. Mirrors the legend/minimap
 * overlay treatment (absolute, top-left), but unlike the purely-informational
 * legend it must accept clicks, so it sets ``pointer-events: auto`` on itself.
 * Hidden entirely while nothing is selected.
 */
@Component({
  selector: 'vt-browse-selection-panel',
  standalone: true,
  imports: [CommonModule],
  template: `
    @if (count > 0) {
      <div class="browse-selection">
        <span class="browse-selection-count">{{ count | number }} selected</span>
        <button
          type="button"
          class="browse-selection-clear"
          (click)="clear()"
          title="Clear selection"
          aria-label="Clear selection">
          Clear
        </button>
      </div>
    }
  `,
  styleUrl: './browse-selection-panel.component.scss',
})
export class BrowseSelectionPanelComponent implements OnInit, OnDestroy {
  /** Mirrored from the selection service so the template re-renders on change. */
  count = 0;
  private sub: Subscription | null = null;

  constructor(private selection: BrowseSelectionService) {}

  ngOnInit(): void {
    this.count = this.selection.size;
    this.sub = this.selection.changed$.subscribe(() => {
      this.count = this.selection.size;
    });
  }

  ngOnDestroy(): void {
    this.sub?.unsubscribe();
  }

  clear(): void {
    this.selection.clear();
  }
}
