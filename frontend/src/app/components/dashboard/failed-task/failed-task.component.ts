import { ChangeDetectionStrategy, Component, computed, input, output } from '@angular/core';

import { LoadingTask } from '../../../models/api.models';
import { formatTimestamp } from '../../../utils/format-date';
import { CopyDetailButtonComponent } from '../../copy-detail-button/copy-detail-button.component';

/**
 * The body of a failed dataset task's dashboard row: the error, what the task
 * was reading and when it failed, a copy-debug-info button, and Dismiss.
 *
 * A failed import never becomes a dataset, so this row is all that is left of
 * it once the failure toast is gone; the backend keeps it until Dismiss
 * (#4973). Extra actions (the HuggingFace sign-in offered on a gated download)
 * are projected in ahead of the copy button.
 */
@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-failed-task',
  standalone: true,
  imports: [CopyDetailButtonComponent],
  template: `
    <div class="failed-task__text">
      <span class="error-msg" [title]="task().error">Failed: {{ task().error }}</span>
      <span class="failed-task__meta">{{ meta() }}</span>
    </div>
    <div class="failed-task__actions">
      <ng-content />
      <vt-copy-detail-button [value]="debugInfo()" label="debug info" />
      <button class="btn btn--cancel btn--sm" (click)="dismiss.emit()" title="Dismiss this error">Dismiss</button>
    </div>
  `,
  styleUrl: './failed-task.component.scss',
})
export class FailedTaskComponent {
  readonly task = input.required<LoadingTask>();
  readonly dismiss = output<void>();

  /** When the task stopped; its start stands in until the finish is stamped. */
  private readonly failedAt = computed(() => this.task().finished_at ?? this.task().created_at);

  /** "<what it read> · failed <when>", or just the time when the source is unknown. */
  readonly meta = computed(() => {
    const when = `failed ${formatTimestamp(this.failedAt())}`;
    const source = this.task().source;
    return source ? `${source} · ${when}` : when;
  });

  /** A markdown block for an issue or a chat, in the shape the error toasts copy. */
  readonly debugInfo = computed(() => {
    const t = this.task();
    const iso = (seconds: number | undefined) => (seconds ? new Date(seconds * 1000).toISOString() : '-');
    const lines = ['**VTSearch error**', '', '- **Message:** Dataset import failed'];
    lines.push(`- **Dataset:** ${t.name}`);
    lines.push(`- **Error:** ${t.error}`);
    if (t.source) lines.push(`- **Source:** ${t.source}`);
    if (t.media_type) lines.push(`- **Media type:** ${t.media_type}`);
    if (t.embedder) lines.push(`- **Embedder:** \`${t.embedder}\``);
    lines.push(`- **Task:** \`${t.task_id}\``);
    lines.push(`- **Started:** ${iso(t.created_at)}`);
    lines.push(`- **Failed:** ${iso(this.failedAt())}`);
    return lines.join('\n');
  });
}
