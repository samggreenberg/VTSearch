import { Injectable } from '@angular/core';

export interface ModelExample {
  type: string;
  value: string;
}

/**
 * Lightweight session state passed from dashboard to label view.
 * Holds the selected model's text_query or media_example so autopilot
 * can auto-sort on entry.  Also carries the full examples list so the
 * label view can seed good votes from media examples.
 */
@Injectable({ providedIn: 'root' })
export class LabelSessionService {
  textQuery = '';
  mediaExample = '';
  examples: ModelExample[] = [];
}
