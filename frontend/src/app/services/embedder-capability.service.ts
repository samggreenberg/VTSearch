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

  /** Capability metadata for *name*, or `undefined` when unknown / unloaded. */
  private infoFor(name: string): EmbedderInfo | undefined {
    const infos = this.infos();
    if (!infos || !name) return undefined;
    return infos.find((e) => e.name === name);
  }

  /**
   * Whether ANY of *names* can embed text queries (the v3 trio: a dataset bound
   * to a text embedder alongside vision-only ones still offers text sort).
   * Defaults to `true` for an empty/unknown list so a working feature is never
   * hidden on missing metadata.
   */
  supportsTextAny(names: readonly string[] | null | undefined): boolean {
    if (!names || names.length === 0) return true;
    return names.some((n) => this.supportsText(n));
  }

  /**
   * Whether ANY of *names* emits a best-match region overlay — patch-region
   * embedders (DINOv2/v3, EUPE) or structural embedders (SIFT/VLAD). Defaults
   * to `false` (unknown / unloaded) so a dead toggle never appears.
   */
  supportsRegionOverlayAny(names: readonly string[] | null | undefined): boolean {
    if (!names) return false;
    return names.some((n) => {
      const info = this.infoFor(n);
      return info?.supports_patch_regions === true || info?.supports_geometric_verification === true;
    });
  }

  /**
   * Whether ANY of *names* is a structural (instance-matching) embedder.
   * Defaults to `false` so the structural marquee copy only shows when a
   * structural embedder is actually bound.
   */
  supportsStructuralAny(names: readonly string[] | null | undefined): boolean {
    if (!names) return false;
    return names.some((n) => this.infoFor(n)?.supports_geometric_verification === true);
  }
}
