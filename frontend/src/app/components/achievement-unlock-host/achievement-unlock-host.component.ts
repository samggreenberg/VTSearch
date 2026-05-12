import { Component, OnDestroy, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Subject } from 'rxjs';
import { takeUntil } from 'rxjs/operators';
import { ModalComponent } from '../modal/modal.component';
import { AchievementBadgeComponent } from '../achievement-badge/achievement-badge.component';
import {
  AchievementsService,
  PendingAnnouncement,
} from '../../services/achievements.service';

/**
 * Global host that listens to `AchievementsService.unlocks` and pops a
 * VtModal for each one — same shell as the "Are you sure?" dialog, but
 * branded with the tier-decorated badge for the unlocked tier.  Multiple
 * unlocks queue and play back one at a time.
 */
@Component({
  selector: 'vt-achievement-unlock-host',
  standalone: true,
  imports: [CommonModule, ModalComponent, AchievementBadgeComponent],
  templateUrl: './achievement-unlock-host.component.html',
  styleUrl: './achievement-unlock-host.component.scss',
})
export class AchievementUnlockHostComponent implements OnInit, OnDestroy {
  current: PendingAnnouncement | null = null;
  private queue: PendingAnnouncement[] = [];
  private destroy$ = new Subject<void>();

  constructor(private achievements: AchievementsService) {}

  ngOnInit(): void {
    this.achievements.unlocks
      .pipe(takeUntil(this.destroy$))
      .subscribe((unlock) => this.enqueue(unlock));
  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
  }

  get open(): boolean {
    return this.current !== null;
  }

  acknowledge(): void {
    if (this.current === null) return;
    const { id, tier_idx } = this.current;
    this.achievements.acknowledge(id, tier_idx);
    this.current = null;
    this.popNext();
  }

  private enqueue(unlock: PendingAnnouncement): void {
    const key = (p: PendingAnnouncement) => `${p.id}:${p.tier_idx}`;
    if (this.current && key(this.current) === key(unlock)) return;
    if (this.queue.some((p) => key(p) === key(unlock))) return;
    this.queue.push(unlock);
    if (this.current === null) {
      this.popNext();
    }
  }

  private popNext(): void {
    this.current = this.queue.shift() ?? null;
  }
}
