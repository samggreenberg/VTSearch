import { Injectable, computed, inject, signal } from '@angular/core';

import { SourcePhotoService } from './source-photo.service';

/**
 * Collapse to photos (#4751): show a face dataset's Train and Tune rankings as
 * the photos the faces were cut from.
 *
 * A presentation mode on the ranking, never a new ranking: the server groups
 * the faces of the ranking on screen by photo (`GET /api/sort/collapsed`), a
 * photo scores as its best face, and every matching face is outlined on it.
 * Votes, counts and the Find actions stay on faces, since the detector is a
 * face detector.
 *
 * The switch is the user's and lasts the session, across Train and Tune and
 * dataset switches; it takes effect only where Show in photo is offered (a
 * convert-in dataset whose import also made an Image dataset), so turning it
 * on for a face dataset leaves every other dataset's list as it was.
 */
@Injectable({ providedIn: 'root' })
export class CollapsePhotosService {
  private readonly sourcePhoto = inject(SourcePhotoService);

  private readonly _on = signal(false);
  /** The user's switch, whether or not the active dataset can collapse. */
  readonly on = this._on.asReadonly();
  /** The active dataset's faces have photos to collapse into. */
  readonly offered = computed(() => this.sourcePhoto.offered());
  /** Collapse the ranking on screen: switched on, on a dataset that can. */
  readonly active = computed(() => this._on() && this.offered());

  toggle(): void {
    this._on.update((on) => !on);
  }

  set(on: boolean): void {
    this._on.set(on);
  }
}
