import { Component, ElementRef, EventEmitter, Input, OnDestroy, Output, ViewChild } from '@angular/core';
import { MediasApiService } from '../../../services/medias-api.service';
import { VoteStateService } from '../../../services/vote-state.service';

@Component({
  selector: 'vt-voting-overlay',
  standalone: true,
  templateUrl: './voting-overlay.component.html',
  styleUrl: './voting-overlay.component.scss',
})
export class VotingOverlayComponent implements OnDestroy {
  @Input() isGood = false;
  @Input() isBad = false;
  @Input() disabled = false;
  @Output() voted = new EventEmitter<'good' | 'bad'>();
  @Output() mediaAdded = new EventEmitter<{ media_id: number; label: 'good' | 'bad'; is_new: boolean }>();

  @ViewChild('addGoodInput') addGoodInput!: ElementRef<HTMLInputElement>;
  @ViewChild('addBadInput') addBadInput!: ElementRef<HTMLInputElement>;

  goodFlash = false;
  badFlash = false;
  addingGood = false;
  addingBad = false;

  private goodTimer: ReturnType<typeof setTimeout> | null = null;
  private badTimer: ReturnType<typeof setTimeout> | null = null;

  constructor(
    private mediasApi: MediasApiService,
    private voteState: VoteStateService,
  ) {}

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

  triggerAddGood(): void {
    this.addGoodInput.nativeElement.click();
  }

  triggerAddBad(): void {
    this.addBadInput.nativeElement.click();
  }

  onFileSelected(event: Event, label: 'good' | 'bad'): void {
    const input = event.target as HTMLInputElement;
    if (!input.files?.length) return;
    const file = input.files[0];
    input.value = '';

    if (label === 'good') {
      this.addingGood = true;
    } else {
      this.addingBad = true;
    }

    this.mediasApi.addToPile(file, label).subscribe({
      next: (result) => {
        this.mediaAdded.emit({ media_id: result.media_id, label, is_new: result.is_new });
        this.voteState.loadVotes();
        if (label === 'good') {
          this.addingGood = false;
        } else {
          this.addingBad = false;
        }
      },
      error: () => {
        if (label === 'good') {
          this.addingGood = false;
        } else {
          this.addingBad = false;
        }
      },
    });
  }

  ngOnDestroy(): void {
    if (this.goodTimer) clearTimeout(this.goodTimer);
    if (this.badTimer) clearTimeout(this.badTimer);
  }
}
