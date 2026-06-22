import { Injectable, inject, signal } from '@angular/core';

import { DatasetsListingsApiService } from './datasets-listings-api.service';
import type { EmbedderInfo } from '../models/api.models';

/**
 * Process-wide cache of embedder capability metadata (the `GET /api/embedders`
 * registry), so callers can answer "can this dataset's embedder search by
 * text?" without each reimplementing the `supports_text` lookup.
 *
 * Vision-only / speech-only encoders (DINOv3, EUPE, AST, Whisper, VideoMAE)
 * report `supports_text === false`; CLIP/SigLIP/CLAP/E5/X-CLIP report `true`.
 * Text sort and Autopilot's text-seed phase only work on the latter.
 *
 * The default everywhere is **text supported** when we don't know (empty
 * embedder name, registry not loaded yet, or an unrecognised name): we never
 * want to hide a working feature on missing metadata.
 */
@Injectable({ providedIn: 'root' })
export class EmbedderCapabilityService {
  private readonly api = inject(DatasetsListingsApiService);

  /** Loaded embedder metadata, or `null` until the registry resolves. */
  readonly infos = signal<EmbedderInfo[] | null>(null);
  private loading = false;

  /** Kick off a one-time load of the embedder registry. Idempotent. */
  ensureLoaded(): void {
    if (this.infos() !== null || this.loading) return;
    this.loading = true;
    this.api.getEmbedders().subscribe({
      next: (list) => {
        this.infos.set(list ?? []);
        this.loading = false;
      },
      error: () => {
        this.infos.set([]);
        this.loading = false;
      },
    });
  }

  /** True once the registry has resolved (success or failure). */
  get loaded(): boolean {
    return this.infos() !== null;
  }

  /**
   * Whether *embedderName* can embed text queries. Defaults to `true` when the
   * name is empty or the registry hasn't loaded / doesn't recognise it, so a
   * working text feature is never hidden on missing metadata.
   */
  supportsText(embedderName: string): boolean {
    if (!embedderName) return true;
    const infos = this.infos();
    if (!infos) return true;
    const info = infos.find((e) => e.name === embedderName);
    return info ? info.supports_text !== false : true;
  }
}
