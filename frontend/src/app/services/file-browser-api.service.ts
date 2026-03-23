import { Injectable } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface BrowseEntry {
  name: string;
  path: string;
  size_bytes?: number;
}

export interface BrowseResponse {
  directories: BrowseEntry[];
  files: BrowseEntry[];
  current_path: string;
  root: string;
}

@Injectable({ providedIn: 'root' })
export class FileBrowserApiService {
  constructor(private http: HttpClient) {}

  browse(path: string, extensions?: string): Observable<BrowseResponse> {
    let params = new HttpParams().set('path', path);
    if (extensions) {
      params = params.set('extensions', extensions);
    }
    return this.http.get<BrowseResponse>('/api/browse', { params });
  }
}
