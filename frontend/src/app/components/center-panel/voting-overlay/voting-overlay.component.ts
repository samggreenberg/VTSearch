import { Component, EventEmitter, Input, Output } from '@angular/core';

@Component({
  selector: 'vt-voting-overlay',
  standalone: true,
  templateUrl: './voting-overlay.component.html',
  styleUrl: './voting-overlay.component.scss',
})
export class VotingOverlayComponent {
  @Input() isGood = false;
  @Input() isBad = false;
  @Input() disabled = false;
  @Output() voted = new EventEmitter<'good' | 'bad'>();

  goodFlash = false;
  badFlash = false;

  onVoteGood(): void {
    if (this.disabled) return;
    this.goodFlash = true;
    this.voted.emit('good');
    setTimeout(() => (this.goodFlash = false), 300);
  }

  onVoteBad(): void {
    if (this.disabled) return;
    this.badFlash = true;
    this.voted.emit('bad');
    setTimeout(() => (this.badFlash = false), 300);
  }
}
