import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { map } from 'rxjs/operators';

import { ApiConfiguration } from '../generated/api-client/api-configuration';
import type { AutorunExtractorCreateRequest } from '../generated/api-client/models/autorun-extractor-create-request';
import type { AutorunExtractorsListResponse } from '../generated/api-client/models/autorun-extractors-list-response';
import type { AutorunLocalizerCreateRequest } from '../generated/api-client/models/autorun-localizer-create-request';
import type { AutorunLocalizersListResponse } from '../generated/api-client/models/autorun-localizers-list-response';
import type { AutorunProcessorCreateResponse } from '../generated/api-client/models/autorun-processor-create-response';
import type { AutorunProcessorDeleteResponse } from '../generated/api-client/models/autorun-processor-delete-response';
import type { AutorunProcessorRenameResponse } from '../generated/api-client/models/autorun-processor-rename-response';
import type { PregenProcessorsAddResponse } from '../generated/api-client/models/pregen-processors-add-response';
import type { PregenProcessorsListResponse } from '../generated/api-client/models/pregen-processors-list-response';
import { addAutorunExtractorRoute } from '../generated/api-client/fn/processors-crud/add-autorun-extractor-route';
import { addAutorunLocalizerRoute } from '../generated/api-client/fn/processors-crud/add-autorun-localizer-route';
import { addPregenProcessors } from '../generated/api-client/fn/processors-crud/add-pregen-processors';
import { deleteAutorunExtractorRoute } from '../generated/api-client/fn/processors-crud/delete-autorun-extractor-route';
import { deleteAutorunLocalizerRoute } from '../generated/api-client/fn/processors-crud/delete-autorun-localizer-route';
import { getAutorunExtractorsRoute } from '../generated/api-client/fn/processors-crud/get-autorun-extractors-route';
import { getAutorunLocalizersRoute } from '../generated/api-client/fn/processors-crud/get-autorun-localizers-route';
import { listPregenProcessors } from '../generated/api-client/fn/processors-crud/list-pregen-processors';
import { renameAutorunExtractorRoute } from '../generated/api-client/fn/processors-crud/rename-autorun-extractor-route';
import { renameAutorunLocalizerRoute } from '../generated/api-client/fn/processors-crud/rename-autorun-localizer-route';

/** Autorun extractor / localizer CRUD plus the pregen-processors list /
 *  bulk-add endpoints. */
@Injectable({ providedIn: 'root' })
export class ProcessorsApiService {
  private http = inject(HttpClient);
  private config = inject(ApiConfiguration);

  // --- Extractors ---

  getAutorunExtractors(): Observable<AutorunExtractorsListResponse> {
    return getAutorunExtractorsRoute(this.http, this.config.rootUrl).pipe(map((r) => r.body));
  }

  createExtractor(
    params: AutorunExtractorCreateRequest,
  ): Observable<AutorunProcessorCreateResponse> {
    return addAutorunExtractorRoute(this.http, this.config.rootUrl, { body: params }).pipe(
      map((r) => r.body),
    );
  }

  deleteExtractor(name: string): Observable<AutorunProcessorDeleteResponse> {
    return deleteAutorunExtractorRoute(this.http, this.config.rootUrl, { name }).pipe(
      map((r) => r.body),
    );
  }

  renameExtractor(name: string, newName: string): Observable<AutorunProcessorRenameResponse> {
    return renameAutorunExtractorRoute(this.http, this.config.rootUrl, {
      name,
      body: { new_name: newName },
    }).pipe(map((r) => r.body));
  }

  // --- Localizers ---

  getAutorunLocalizers(): Observable<AutorunLocalizersListResponse> {
    return getAutorunLocalizersRoute(this.http, this.config.rootUrl).pipe(map((r) => r.body));
  }

  createLocalizer(
    params: AutorunLocalizerCreateRequest,
  ): Observable<AutorunProcessorCreateResponse> {
    return addAutorunLocalizerRoute(this.http, this.config.rootUrl, { body: params }).pipe(
      map((r) => r.body),
    );
  }

  deleteLocalizer(name: string): Observable<AutorunProcessorDeleteResponse> {
    return deleteAutorunLocalizerRoute(this.http, this.config.rootUrl, { name }).pipe(
      map((r) => r.body),
    );
  }

  renameLocalizer(name: string, newName: string): Observable<AutorunProcessorRenameResponse> {
    return renameAutorunLocalizerRoute(this.http, this.config.rootUrl, {
      name,
      body: { new_name: newName },
    }).pipe(map((r) => r.body));
  }

  // --- Pregen processors ---

  getPregenProcessors(): Observable<PregenProcessorsListResponse> {
    return listPregenProcessors(this.http, this.config.rootUrl).pipe(map((r) => r.body));
  }

  addPregenProcessors(): Observable<PregenProcessorsAddResponse> {
    return addPregenProcessors(this.http, this.config.rootUrl).pipe(map((r) => r.body));
  }
}
