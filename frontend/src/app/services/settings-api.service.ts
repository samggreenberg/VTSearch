import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { map } from 'rxjs/operators';

import { ApiConfiguration } from '../generated/api-client/api-configuration';
import type { AppSettings } from '../generated/api-client/models/app-settings';
import type { SettingsUpdate } from '../generated/api-client/models/settings-update';
import { apiSettingsDefaultsGet } from '../generated/api-client/fn/settings/api-settings-defaults-get';
import { apiSettingsGet } from '../generated/api-client/fn/settings/api-settings-get';
import { apiSettingsPut } from '../generated/api-client/fn/settings/api-settings-put';
import { apiVersionGet } from '../generated/api-client/fn/main/api-version-get';

@Injectable({ providedIn: 'root' })
export class SettingsApiService {
  private http = inject(HttpClient);
  private config = inject(ApiConfiguration);

  getSettings(): Observable<AppSettings> {
    return apiSettingsGet(this.http, this.config.rootUrl).pipe(map((r) => r.body));
  }

  updateSettings(data: SettingsUpdate): Observable<AppSettings> {
    return apiSettingsPut(this.http, this.config.rootUrl, { body: data }).pipe(map((r) => r.body));
  }

  getDefaults(): Observable<AppSettings> {
    return apiSettingsDefaultsGet(this.http, this.config.rootUrl).pipe(map((r) => r.body));
  }

  getVersion(): Observable<{ version: string }> {
    return apiVersionGet(this.http, this.config.rootUrl).pipe(map((r) => r.body));
  }
}
