import { Injectable, inject } from '@angular/core';
import { HttpClient, HttpContext } from '@angular/common/http';
import { Observable } from 'rxjs';

import { ApiConfiguration } from '../generated/api-client/api-configuration';
import type { DetectorRegistryEntry } from '../generated/api-client/models/detector-registry-entry';
import type { DetectorSharingIntegrity } from '../generated/api-client/models/detector-sharing-integrity';
import { SKIP_ERROR_TOAST } from '../interceptors/error.interceptor';

// INTERIM (#5578 ahead of #5577): these shapes and URLs are #5577's documented
// contract, typed by hand until its OpenAPI snapshot lands; then every method
// wraps the generated function and these interfaces go.

/** `GET /api/detectors/sharing`: whether sharing is on, and through what. */
export interface SharingCapability {
  enabled: boolean;
  name?: string | null;
  display_name?: string | null;
  icon?: string | null;
  supports_recipient_search?: boolean;
}

/** What a container's foreign records get on a receive or refresh. */
export type ForeignPolicy = 'abort' | 'import' | 'skip';

/** `POST .../refresh`: applied or not, what the container held, and the entry after. */
export interface SharingRefreshResponse {
  applied: boolean;
  integrity: DetectorSharingIntegrity;
  entry?: DetectorRegistryEntry;
  message?: string;
}

/** `POST .../fork`: the new entry, and a note when the copy could not be shared. */
export interface SharingForkResponse {
  entry: DetectorRegistryEntry;
  remote_sync?: string | null;
  message?: string;
}

/** Detector sharing: the capability, Check for updates, Fork. */
@Injectable({ providedIn: 'root' })
export class DetectorsSharingApiService {
  private http = inject(HttpClient);
  private config = inject(ApiConfiguration);

  private url(path: string): string {
    return `${this.config.rootUrl}${path}`;
  }

  /** Never toasts: a server without the route is a server without sharing. */
  getCapability(): Observable<SharingCapability> {
    return this.http.get<SharingCapability>(this.url('/api/detectors/sharing'), {
      context: new HttpContext().set(SKIP_ERROR_TOAST, true),
    });
  }

  refresh(detectorId: string, foreign: ForeignPolicy): Observable<SharingRefreshResponse> {
    return this.http.post<SharingRefreshResponse>(
      this.url(`/api/detectors/registry/${encodeURIComponent(detectorId)}/refresh`),
      { foreign },
    );
  }

  fork(detectorId: string, name?: string): Observable<SharingForkResponse> {
    return this.http.post<SharingForkResponse>(
      this.url(`/api/detectors/registry/${encodeURIComponent(detectorId)}/fork`),
      name ? { name } : {},
    );
  }
}
