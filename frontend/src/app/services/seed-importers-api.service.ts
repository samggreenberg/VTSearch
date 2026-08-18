import { HttpClient } from '@angular/common/http';
import { inject, Injectable } from '@angular/core';
import { Observable } from 'rxjs';

import { FieldOptions, ImporterInfo } from '../models/api.models';

/** Response of ``GET /api/seed-importers``.
 *
 *  No `tabs`: each seed importer *is* its own tab in the New Detector
 *  modal's Blank flow, so there is no shared category bar to declare. */
export interface SeedImportersResponse {
  importers: ImporterInfo[];
}

/** One unlabeled seed saved by ``POST /api/seed-import/<name>``. */
export interface SeedImportItem {
  filename: string;
  original_name: string;
  /** Durable origin dict (`{importer, params}`) when the seed has a
   *  re-fetchable identity; `null` when the saved bytes are the only
   *  record.  Persisted on the detector example, same as a datasource
   *  importer's single fetch. */
  origin?: Record<string, unknown> | null;
}

/** Response of ``POST /api/seed-import/<name>``. */
export interface SeedImportResult {
  items: SeedImportItem[];
  count: number;
  /** True when the importer returned more than its `max_items` cap and the
   *  tail was dropped server-side. */
  truncated: boolean;
}

/** API client for seed importers: plugins that contribute a *batch* of
 *  unlabeled seed media ("close but not quite" examples) to a new blank
 *  detector.  Unlike a datasource importer's single hand-picked exemplar,
 *  a seed is a query hint, not a Good vote — the detector stores it with
 *  `labeled: false`.
 *
 *  Uses plain ``HttpClient`` because the run endpoint's body shape is
 *  plugin-dependent (same convention as {@link DatasourceImportersApiService}).
 *  Structurally satisfies `PluginImportApi`, so `<vt-plugin-import-form>`
 *  can render any seed importer's fields with no family-specific code. */
@Injectable({ providedIn: 'root' })
export class SeedImportersApiService {
  private http = inject(HttpClient);

  /** List the available seed importers.  Empty on a vanilla install: the
   *  family ships no built-ins, so the Blank flow shows only its stock
   *  Text and media tabs until a plugin registers one. */
  list(): Observable<SeedImportersResponse> {
    return this.http.get<SeedImportersResponse>('/api/seed-importers');
  }

  /** Run the named seed importer with the given form-field values.  When
   *  the importer declares a ``file`` field, pass the picked ``File`` and
   *  its field key so the request goes out as multipart. */
  run(
    importerName: string,
    values: Record<string, string>,
    file?: File,
    fileFieldKey?: string,
  ): Observable<SeedImportResult> {
    const url = `/api/seed-import/${encodeURIComponent(importerName)}`;
    if (file && fileFieldKey) {
      const formData = new FormData();
      formData.append(fileFieldKey, file, file.name);
      for (const [key, value] of Object.entries(values)) {
        if (key !== fileFieldKey) {
          formData.append(key, String(value ?? ''));
        }
      }
      return this.http.post<SeedImportResult>(url, formData);
    }
    return this.http.post<SeedImportResult>(url, values);
  }

  /** Fetch the current options for a ``dynamic_options`` select field. */
  getFieldOptions(
    importerName: string,
    fieldKey: string,
    values: Record<string, string>,
  ): Observable<{ options: FieldOptions[] }> {
    return this.http.post<{ options: FieldOptions[] }>(
      `/api/seed-import/${encodeURIComponent(importerName)}/options`,
      { field_key: fieldKey, values },
    );
  }
}
