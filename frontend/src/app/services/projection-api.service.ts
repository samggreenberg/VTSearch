import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { map } from 'rxjs/operators';
import type { ProjectionMeta, ProjectionBuildResponse, TilePayload } from '../models/projection.models';

@Injectable({ providedIn: 'root' })
export class ProjectionApiService {
  private http = inject(HttpClient);

  getMeta(): Observable<ProjectionMeta> {
    return this.http.get<ProjectionMeta>('/api/projection/meta');
  }

  build(): Observable<ProjectionBuildResponse> {
    return this.http.post<ProjectionBuildResponse>('/api/projection/build', {});
  }

  getTile(projectionId: string, level: number, tx: number, ty: number): Observable<TilePayload> {
    return this.http.get<TilePayload>(
      `/api/projection/tiles/${projectionId}/${level}/${tx}/${ty}`,
    );
  }
}
