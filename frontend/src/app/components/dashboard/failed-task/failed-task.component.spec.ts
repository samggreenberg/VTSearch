import { ComponentFixture, TestBed } from '@angular/core/testing';

import { FailedTaskComponent } from './failed-task.component';
import { LoadingTask } from '../../../models/api.models';
import { configureZoneless } from '../../../testing/zoneless-testbed';
import { settleZoneless } from '../../../testing/settle-resource';
import { formatTimestamp } from '../../../utils/format-date';

describe('FailedTaskComponent', () => {
  let fixture: ComponentFixture<FailedTaskComponent>;

  const FINISHED = 1_790_000_000;
  const failed: LoadingTask = {
    status: 'idle',
    message: '',
    current: 0,
    total: 0,
    task_id: '_loading_604ba3b7',
    name: 'Doomed import',
    created_at: FINISHED - 5,
    finished_at: FINISHED,
    error: 'Import produced no medias.',
    source: 'server_folder:/data/empty',
    media_type: 'image',
  };

  async function render(task: LoadingTask): Promise<HTMLElement> {
    fixture.componentRef.setInput('task', task);
    await settleZoneless(fixture);
    return fixture.nativeElement as HTMLElement;
  }

  beforeEach(async () => {
    await configureZoneless({ imports: [FailedTaskComponent] }).compileComponents();
    fixture = TestBed.createComponent(FailedTaskComponent);
  });

  it('shows the error, then what the task read and when it failed', async () => {
    const el = await render(failed);
    expect(el.querySelector('.error-msg')?.textContent).toBe('Failed: Import produced no medias.');
    expect(el.querySelector('.failed-task__meta')?.textContent).toBe(
      `server_folder:/data/empty · failed ${formatTimestamp(FINISHED)}`,
    );
  });

  it('falls back to the start time and drops the source when either is missing', async () => {
    const el = await render({ ...failed, source: undefined, finished_at: undefined });
    expect(el.querySelector('.failed-task__meta')?.textContent).toBe(`failed ${formatTimestamp(FINISHED - 5)}`);
  });

  it('builds debug info naming the task, its source and the error', async () => {
    await render(failed);
    const info = fixture.componentInstance.debugInfo();
    expect(info).toContain('- **Error:** Import produced no medias.');
    expect(info).toContain('- **Source:** server_folder:/data/empty');
    expect(info).toContain('- **Task:** `_loading_604ba3b7`');
    expect(info).toContain(`- **Failed:** ${new Date(FINISHED * 1000).toISOString()}`);
  });

  it('emits dismiss from the Dismiss button', async () => {
    const el = await render(failed);
    const dismissed = vi.fn();
    fixture.componentInstance.dismiss.subscribe(dismissed);
    const button = [...el.querySelectorAll('button')].find((b) => b.textContent?.trim() === 'Dismiss');
    button!.click();
    expect(dismissed).toHaveBeenCalledTimes(1);
  });
});
