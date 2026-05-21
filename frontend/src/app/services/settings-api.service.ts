import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { map } from 'rxjs/operators';

import { ApiConfiguration } from '../generated/api-client/api-configuration';
import type { AppSettings } from '../generated/api-client/models/app-settings';
import type { SettingsUpdate } from '../generated/api-client/models/settings-update';
import { getDefaults } from '../generated/api-client/fn/settings/get-defaults';
import { getSettings } from '../generated/api-client/fn/settings/get-settings';
import { updateSettings } from '../generated/api-client/fn/settings/update-settings';
import { version } from '../generated/api-client/fn/main/version';

@Injectable({ providedIn: 'root' })
export class SettingsApiService {
  private http = inject(HttpClient);
  private config = inject(ApiConfiguration);

  getSettings(): Observable<AppSettings> {
    return getSettings(this.http, this.config.rootUrl).pipe(map((r) => r.body));
  }

  updateSettings(data: SettingsUpdate): Observable<AppSettings> {
    return updateSettings(this.http, this.config.rootUrl, { body: data }).pipe(map((r) => r.body));
  }

  getDefaults(): Observable<AppSettings> {
    return getDefaults(this.http, this.config.rootUrl).pipe(map((r) => r.body));
  }

  getVersion(): Observable<{ version: string }> {
    return version(this.http, this.config.rootUrl).pipe(map((r) => r.body));
  }
}
