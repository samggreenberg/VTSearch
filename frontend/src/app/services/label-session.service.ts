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
  modelName = '';

  /** Server-side filenames of every media example, for Autopilot's example
   *  sort (plural examples rank against their embedding centroid).  Falls
   *  back to the scalar `mediaExample` for detectors whose registry entry
   *  predates the `examples` list. */
  get mediaExampleFilenames(): string[] {
    const fromList = this.examples
      .filter((e) => e.type === 'media' && e.value)
      .map((e) => e.value);
    if (fromList.length > 0) return fromList;
    return this.mediaExample ? [this.mediaExample] : [];
  }

  /** Total votes cast since the last re-sort prompt. */
  resortVoteCount = 0;

  /** Next threshold at which to show the re-sort prompt. */
  resortNextThreshold = 0;
}
