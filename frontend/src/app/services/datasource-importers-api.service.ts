import { HttpClient } from '@angular/common/http';
import { inject, Injectable } from '@angular/core';
import { Observable } from 'rxjs';

import { FieldOptions, ImporterInfo, ImporterPickerTab } from '../models/api.models';

/** Response of ``GET /api/datasource-importers``. */
export interface DatasourceImportersResponse {
  importers: ImporterInfo[];
  tabs: ImporterPickerTab[];
}

/** Response of ``POST /api/datasource-import/<name>`` (the example-media
 *  upload contract plus the fetched item's durable origin). */
export interface DatasourceImportResult {
  filename: string;
  original_name: string;
  /** Durable origin dict (`{importer, params}`) when the importer's items
   *  have a re-fetchable identity (a URL, a server path); persisted on the
   *  detector example so the seed survives the example_media/ cache file.
   *  `null` for items with no re-derivable source. */
  origin?: Record<string, unknown> | null;
}

/** API client for datasource importers: plugins that fetch a *single*
 *  media item (from a URL, a server path, a third-party service) into
 *  the server-side example-media directory.
 *
 *  Uses plain ``HttpClient`` because the run endpoint's body shape is
 *  plugin-dependent (same convention as ``runImporter`` in
 *  {@link DatasetsCrudApiService}). */
@Injectable({ providedIn: 'root' })
export class DatasourceImportersApiService {
  private http = inject(HttpClient);

  /** List available datasource importers plus the shared picker tabs. */
  list(): Observable<DatasourceImportersResponse> {
    return this.http.get<DatasourceImportersResponse>('/api/datasource-importers');
  }

  /** Run the named importer with the given form-field values.  When the
   *  importer declares a ``file`` field, pass the picked ``File`` and its
   *  field key so the request goes out as multipart. */
  run(
    importerName: string,
    values: Record<string, string>,
    file?: File,
    fileFieldKey?: string,
  ): Observable<DatasourceImportResult> {
    const url = `/api/datasource-import/${encodeURIComponent(importerName)}`;
    if (file && fileFieldKey) {
      const formData = new FormData();
      formData.append(fileFieldKey, file, file.name);
      for (const [key, value] of Object.entries(values)) {
        if (key !== fileFieldKey) {
          formData.append(key, String(value ?? ''));
        }
      }
      return this.http.post<DatasourceImportResult>(url, formData);
    }
    return this.http.post<DatasourceImportResult>(url, values);
  }

  /** Fetch the current options for a ``dynamic_options`` select field. */
  getFieldOptions(
    importerName: string,
    fieldKey: string,
    values: Record<string, string>,
  ): Observable<{ options: FieldOptions[] }> {
    return this.http.post<{ options: FieldOptions[] }>(
      `/api/datasource-import/${encodeURIComponent(importerName)}/options`,
      { field_key: fieldKey, values },
    );
  }
}
