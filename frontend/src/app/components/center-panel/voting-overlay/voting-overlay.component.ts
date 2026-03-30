import { Component, EventEmitter, Input, OnDestroy, Output } from '@angular/core';
import { IconComponent } from '../../icon/icon.component';

@Component({
  selector: 'vt-voting-overlay',
  standalone: true,
  imports: [IconComponent],
  templateUrl: './voting-overlay.component.html',
  styleUrl: './voting-overlay.component.scss',
})
export class VotingOverlayComponent implements OnDestroy {
  @Input() isGood = false;
  @Input() isBad = false;
  @Input() disabled = false;
  @Output() voted = new EventEmitter<'good' | 'bad'>();

  goodFlash = false;
  badFlash = false;

  private goodTimer: ReturnType<typeof setTimeout> | null = null;
  private badTimer: ReturnType<typeof setTimeout> | null = null;

  onVoteGood(): void {
    if (this.disabled) return;
    this.goodFlash = true;
    this.voted.emit('good');
    if (this.goodTimer) clearTimeout(this.goodTimer);
    this.goodTimer = setTimeout(() => (this.goodFlash = false), 300);
  }

  onVoteBad(): void {
    if (this.disabled) return;
    this.badFlash = true;
    this.voted.emit('bad');
    if (this.badTimer) clearTimeout(this.badTimer);
    this.badTimer = setTimeout(() => (this.badFlash = false), 300);
  }

  ngOnDestroy(): void {
    if (this.goodTimer) clearTimeout(this.goodTimer);
    if (this.badTimer) clearTimeout(this.badTimer);
  }
}
