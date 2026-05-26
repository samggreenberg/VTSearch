import { Component, OnDestroy, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Subject } from 'rxjs';
import { takeUntil } from 'rxjs/operators';
import { AchievementBadgeComponent } from '../achievement-badge/achievement-badge.component';
import { AchievementsService } from '../../services/achievements.service';
import { SettingsStateService } from '../../services/settings-state.service';
import type { AchievementEntry } from '../../generated/api-client/models/achievement-entry';
import type { AchievementState } from '../../generated/api-client/models/achievement-state';
import type { DocEntry } from '../../generated/api-client/models/doc-entry';

const TIER_NAMES = ['Bronze', 'Silver', 'Gold', 'Platinum'];

interface AchievementRow extends AchievementEntry {
  tierName: string;
  prevThreshold: number;
  progressPct: number;
  progressLabel: string;
  lockReason: string;
}

@Component({
  selector: 'vt-achievements-tab',
  standalone: true,
  imports: [CommonModule, FormsModule, AchievementBadgeComponent],
  templateUrl: './achievements-tab.component.html',
  styleUrl: './achievements-tab.component.scss',
})
export class AchievementsTabComponent implements OnInit, OnDestroy {
  rows: AchievementRow[] = [];
  docs: DocEntry[] = [];
  loading = true;
  totalScore = 0;

  docsExpanded = false;
  phraseInput = '';
  phraseStatus: { kind: 'idle' | 'success' | 'already' | 'wrong'; message: string } = {
    kind: 'idle',
    message: '',
  };
  submitting = false;

  private destroy$ = new Subject<void>();
  private disableAchievements = false;
  private lastState: AchievementState | null = null;

  constructor(
    private achievements: AchievementsService,
    private settingsState: SettingsStateService,
  ) {}

  ngOnInit(): void {
    this.achievements.state.pipe(takeUntil(this.destroy$)).subscribe((state) => {
      this.lastState = state;
      this.applyState(state);
    });
    this.settingsState.settings$.pipe(takeUntil(this.destroy$)).subscribe((s) => {
      this.disableAchievements = !!s?.disable_achievements;
      if (this.lastState) this.applyState(this.lastState);
    });
    this.achievements.refresh();
  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
  }

  toggleDocsExpanded(): void {
    this.docsExpanded = !this.docsExpanded;
  }

  docRawUrl(docId: string): string {
    return this.achievements.docRawUrl(docId);
  }

  submitPhrase(): void {
    const phrase = this.phraseInput.trim();
    if (!phrase || this.submitting) return;
    this.submitting = true;
    this.achievements.checkPhrase(phrase).subscribe((result) => {
      this.submitting = false;
      if (result.matched && !result.already_read) {
        this.phraseStatus = {
          kind: 'success',
          message: `Correct! Credit applied for "${result.doc_name}".`,
        };
        this.phraseInput = '';
      } else if (result.matched && result.already_read) {
        this.phraseStatus = {
          kind: 'already',
          message: `Already credited for "${result.doc_name}". Try a different doc!`,
        };
      } else {
        this.phraseStatus = {
          kind: 'wrong',
          message: 'No match. Check spelling and try again.',
        };
      }
    });
  }

  private applyState(state: AchievementState): void {
    this.rows = state.achievements.map((a) => this.toRow(a));
    this.docs = state.docs;
    this.loading = state.achievements.length === 0;
    this.totalScore = state.achievements.reduce(
      (sum, a) => sum + (a.tier_idx >= 0 ? 1 << a.tier_idx : 0),
      0,
    );
  }

  private toRow(a: AchievementEntry): AchievementRow {
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
      progressLabel = `${a.counter.toLocaleString()} (maxed out)`;
    }
    return {
      ...a,
      tierName,
      prevThreshold,
      progressPct,
      progressLabel,
      lockReason: a.tier_idx === -1 ? this.lockReason(a) : '',
    };
  }

  private lockReason(a: AchievementEntry): string {
    if (this.disableAchievements) {
      return 'Achievements are disabled in Settings; counters are frozen at zero.';
    }
    const firstTier = a.tiers[0];
    const base = `${a.counter.toLocaleString()} so far - reach ${firstTier.toLocaleString()} to unlock Bronze.`;
    if (a.id === 'datasets_loaded' && a.counter === 0) {
      return `${base} Demo and synthetic dataset loads don't count.`;
    }
    if (a.id === 'docs_read') {
      return `${base} Find the code phrase at the bottom of a doc page and paste it in below.`;
    }
    return base;
  }
}
