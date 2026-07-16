import { ChangeDetectionStrategy, Component, inject, input, linkedSignal, output } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Media } from '../../../models/api.models';
import { ActiveContextService } from '../../../services/active-context.service';
import { MediaTypeCapabilityService } from '../../../services/media-type-capability.service';

@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-media-item',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './media-item.component.html',
  styleUrl: './media-item.component.scss',
})
export class MediaItemComponent {
  private activeContext = inject(ActiveContextService);
  private mediaTypeCaps = inject(MediaTypeCapabilityService);

  readonly media = input.required<Media>();
  readonly active = input(false);
  readonly voteLabel = input<'good' | 'bad' | null>(null);
  readonly score = input<number | null>(null);
  readonly focusMode = input<'click' | 'hover'>('click');

  readonly select = output<number>();
  readonly vote = output<{
    id: number;
    vote: 'good' | 'bad';
}>();
  readonly contextRequest = output<{
    id: number;
    x: number;
    y: number;
}>();

  /** Resets to ``false`` whenever the card is recycled for a different media. */
  private readonly thumbnailFailed = linkedSignal<number, boolean>({
    source: () => this.media().id,
    computation: () => false,
  });

  get thumbnailUrl(): string | null {
    if (this.thumbnailFailed()) return null;
    if (this.mediaTypeCaps.usesThumbnails(this.media().media_type)) {
      // Use the downscaled thumbnail endpoint, not the full-resolution
      // ``/image`` route: a grid of hundreds of high-res items would otherwise
      // force the browser to decode every full-size bitmap at once and exhaust
      // memory. The same thumbnail is reused at every zoom level.
      return this.activeContext.mediaUrl(`/api/medias/${this.media().id}/thumbnail`);
    }
    return null;
  }

  /** True when {@link thumbnailUrl} is an audio waveform — a theme-agnostic
   *  alpha-mask PNG (issue #2369) tinted via a CSS mask, not a plain <img>. */
  get isAudioThumbnail(): boolean {
    return !!this.thumbnailUrl && this.media().media_type === 'audio';
  }

  get placeholderIcon(): string | null {
    if (this.thumbnailUrl) return null;
    if (this.media().media_type === 'audio') return '♫';
    if (this.media().media_type === 'text') return '¶';
    return '□';
  }

  onThumbnailError(): void {
    this.thumbnailFailed.set(true);
  }

  get displayName(): string {
    const media = this.media();
    return media.filename || media.description || `#${media.id}`;
  }

  onClick(): void {
    if (this.focusMode() === 'hover') {
      this.vote.emit({ id: this.media().id, vote: 'bad' });
    } else {
      this.select.emit(this.media().id);
    }
  }

  onContextMenu(event: MouseEvent): void {
    if (this.focusMode() === 'hover') {
      // Hover mode keeps the existing right-click = vote-good shortcut; the
      // context menu is intentionally not available so speed-labeling stays
      // fast. Users can still seed a detector via the dashboard.
      event.preventDefault();
      this.vote.emit({ id: this.media().id, vote: 'good' });
      return;
    }
    event.preventDefault();
    this.contextRequest.emit({ id: this.media().id, x: event.clientX, y: event.clientY });
  }

  onMouseEnter(): void {
    if (this.focusMode() === 'hover') {
      this.select.emit(this.media().id);
    }
  }
}
