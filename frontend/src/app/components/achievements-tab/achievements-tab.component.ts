import { Component, OnDestroy, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Subject } from 'rxjs';
import { takeUntil } from 'rxjs/operators';
import { AchievementBadgeComponent } from '../achievement-badge/achievement-badge.component';
import {
  AchievementInfo,
  AchievementsService,
  AchievementsState,
} from '../../services/achievements.service';

const TIER_NAMES = ['Bronze', 'Silver', 'Gold', 'Platinum'];

interface AchievementRow extends AchievementInfo {
  tierName: string;
  prevThreshold: number;
  progressPct: number;
  progressLabel: string;
}

@Component({
  selector: 'vt-achievements-tab',
  standalone: true,
  imports: [CommonModule, AchievementBadgeComponent],
  templateUrl: './achievements-tab.component.html',
  styleUrl: './achievements-tab.component.scss',
})
export class AchievementsTabComponent implements OnInit, OnDestroy {
  rows: AchievementRow[] = [];
  loading = true;

  private destroy$ = new Subject<void>();

  constructor(private achievements: AchievementsService) {}

  ngOnInit(): void {
    this.achievements.state
      .pipe(takeUntil(this.destroy$))
      .subscribe((state) => this.applyState(state));
    this.achievements.refresh();
  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
  }

  private applyState(state: AchievementsState): void {
    this.rows = state.achievements.map((a) => this.toRow(a));
    this.loading = state.achievements.length === 0;
  }

  private toRow(a: AchievementInfo): AchievementRow {
    const tierName = a.tier_idx >= 0 ? TIER_NAMES[a.tier_idx] : 'Locked';
    const prevThreshold = a.tier_idx >= 0 ? a.tiers[a.tier_idx] : 0;
    let progressPct = 100;
    let progressLabel = '';
    if (a.next_threshold !== null) {
      const span = Math.max(1, a.next_threshold - prevThreshold);
      const filled = Math.max(0, a.counter - prevThreshold);
      progressPct = Math.min(100, Math.round((filled / span) * 100));
      const nextTierName = TIER_NAMES[a.tier_idx + 1] ?? 'Next';
      progressLabel = `${a.counter.toLocaleString()} / ${a.next_threshold.toLocaleString()} → ${nextTierName}`;
    } else {
      progressLabel = `${a.counter.toLocaleString()} — maxed out`;
    }
    return {
      ...a,
      tierName,
      prevThreshold,
      progressPct,
      progressLabel,
    };
  }
}
