import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';
import { IconComponent } from '../icon/icon.component';

/**
 * Renders an achievement icon with a tier-specific decoration layer:
 * - Bronze (0): single circle, brown.
 * - Silver (1): double circle, light-gray.
 * - Gold (2): single circle + laurel branches, gold.
 * - Platinum (3): single circle + sparkle accents, white.
 * - Locked (-1): silhouette icon, dim.
 *
 * The base symbol comes from the existing vt-icon set via *iconType*.
 */
@Component({
  selector: 'vt-achievement-badge',
  standalone: true,
  imports: [CommonModule, IconComponent],
  templateUrl: './achievement-badge.component.html',
  styleUrl: './achievement-badge.component.scss',
})
export class AchievementBadgeComponent {
  @Input() iconType = 'trophy';
  @Input() tierIdx = -1;
  /** Outer badge size in pixels (the inner icon is auto-scaled). */
  @Input() size = 56;

  get tierClass(): string {
    switch (this.tierIdx) {
      case 0:
        return 'tier-bronze';
      case 1:
        return 'tier-silver';
      case 2:
        return 'tier-gold';
      case 3:
        return 'tier-platinum';
      default:
        return 'tier-locked';
    }
  }

  get innerIconSize(): number {
    return Math.max(12, Math.round(this.size * 0.55));
  }

  get showOuterRing(): boolean {
    return this.tierIdx >= 0;
  }

  get showDoubleRing(): boolean {
    return this.tierIdx >= 1;
  }

  get showLaurel(): boolean {
    return this.tierIdx >= 2;
  }

  get showSparkles(): boolean {
    return this.tierIdx >= 3;
  }
}
