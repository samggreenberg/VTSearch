import { ChangeDetectionStrategy, Component, DestroyRef, inject, OnInit } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { AchievementsService } from '../../services/achievements.service';
import { ToastService } from '../../services/toast.service';

@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-achievement-unlock-host',
  standalone: true,
  imports: [],
  template: '',
})
export class AchievementUnlockHostComponent implements OnInit {
  private achievements = inject(AchievementsService);
  private toast = inject(ToastService);

  private readonly destroyRef = inject(DestroyRef);

  ngOnInit(): void {
    this.achievements.unlocks.pipe(takeUntilDestroyed(this.destroyRef)).subscribe((p) => {
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
}
