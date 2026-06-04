import {
  Component,
  ElementRef,
  Input,
  OnChanges,
  OnDestroy,
  SimpleChanges,
  ViewChild,
} from '@angular/core';
import { CommonModule } from '@angular/common';
import { ActiveContextService } from '../../services/active-context.service';
import type { HexHoverEvent } from '../browse-canvas/browse-canvas.component';

@Component({
  selector: 'vt-browse-hover-preview',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './browse-hover-preview.component.html',
  styleUrl: './browse-hover-preview.component.scss',
})
export class BrowseHoverPreviewComponent implements OnChanges, OnDestroy {
  @Input() hover: HexHoverEvent | null = null;
  @Input() mediaType = '';
  @ViewChild('audioEl') audioRef?: ElementRef<HTMLAudioElement>;

  visible = false;
  left = 0;
  top = 0;
  audioSrc = '';
  textContent = '';
  count = 0;
  private textLoadAbort: AbortController | null = null;

  constructor(private activeContext: ActiveContextService) {}

  ngOnChanges(changes: SimpleChanges): void {
    if (changes['hover']) {
      if (this.hover) {
        this.show(this.hover);
      } else {
        this.hide();
      }
    }
  }

  ngOnDestroy(): void {
    this.stopAudio();
    this.textLoadAbort?.abort();
  }

  private show(event: HexHoverEvent): void {
    const representativeId = event.cell.rep_id;

    // Image and video paint their thumbnail directly onto the hex (see
    // browse-canvas); nothing happens on hover, so suppress the pop-up.
    if (this.mediaType === 'image' || this.mediaType === 'video') {
      this.hide();
      return;
    }

    this.visible = true;
    this.left = event.screenX + 16;
    this.top = event.screenY - 8;
    this.count = event.cell.count;

    switch (this.mediaType) {
      case 'audio':
        this.textContent = '';
        this.playAudio(representativeId);
        break;
      case 'text':
        this.stopAudio();
        this.loadText(representativeId);
        break;
      default:
        this.stopAudio();
        this.textContent = `Item #${representativeId}`;
    }
  }

  private hide(): void {
    this.visible = false;
    this.stopAudio();
    this.textLoadAbort?.abort();
    this.textLoadAbort = null;
    this.textContent = '';
  }

  private playAudio(mediaId: number): void {
    const src = this.activeContext.mediaUrl(`/api/medias/${mediaId}/audio`);
    if (this.audioSrc === src) return;
    this.audioSrc = src;

    setTimeout(() => {
      const el = this.audioRef?.nativeElement;
      if (!el) return;
      el.loop = true;
      el.load();
      el.play().catch(() => {});
    });
  }

  private stopAudio(): void {
    const el = this.audioRef?.nativeElement;
    if (el) {
      el.pause();
      el.currentTime = 0;
    }
    this.audioSrc = '';
  }

  private loadText(mediaId: number): void {
    this.textContent = `Loading...`;
    // Cancel any in-flight text load so a slow earlier response can't clobber
    // the preview after the cursor has moved to another hex.
    this.textLoadAbort?.abort();
    const abort = new AbortController();
    this.textLoadAbort = abort;
    const url = this.activeContext.mediaUrl(`/api/medias/${mediaId}/paragraph`);
    fetch(url, { signal: abort.signal })
      .then((r) => r.json())
      .then((data) => {
        if (this.hover?.cell.rep_id === mediaId) {
          const text: string = data.content || '';
          this.textContent = text.length > 300 ? text.slice(0, 300) + '...' : text;
        }
      })
      .catch((err) => {
        if (err?.name === 'AbortError') return;
        if (this.hover?.cell.rep_id === mediaId) {
          this.textContent = `Item #${mediaId}`;
        }
      });
  }
}
