import { Injectable } from '@angular/core';

/**
 * Lightweight session state passed from dashboard to label view.
 * Holds the selected model's text_query so autopilot can auto-sort on entry.
 */
@Injectable({ providedIn: 'root' })
export class LabelSessionService {
  textQuery = '';
}
