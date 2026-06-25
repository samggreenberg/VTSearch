import { ChangeDetectionStrategy, Component, inject, OnDestroy, OnInit } from '@angular/core';
import { Subject } from 'rxjs';
import { takeUntil } from 'rxjs/operators';
import { AchievementsService } from '../../services/achievements.service';
import { ToastService } from '../../services/toast.service';

@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-achievement-unlock-host',
  standalone: true,
  imports: [],
  template: '',
})
export class AchievementUnlockHostComponent implements OnInit, OnDestroy {
  private achievements = inject(AchievementsService);
  private toast = inject(ToastService);

  private destroy$ = new Subject<void>();

  ngOnInit(): void {
    this.achievements.unlocks.pipe(takeUntil(this.destroy$)).subscribe((p) => {
      this.toast.success({
        message: `${p.tier_name}: ${p.name}`,
        detail: `Milestone reached: ${p.threshold.toLocaleString()}`,
        dedupKey: `achievement:${p.id}:${p.tier_idx}`,
        action: {
          label: 'View',
          onClick: () => this.achievements.requestOpenPanel(),
        },
      });
    });
  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
  }
}
