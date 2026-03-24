import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import {
  MediaItem,
  TextResponse,
  VoteResponse,
} from '../models/api.models';

@Injectable({ providedIn: 'root' })
export class MediasApiService {
  constructor(private http: HttpClient) {}

  getMedias(): Observable<MediaItem[]> {
    return this.http.get<MediaItem[]>('/api/medias');
  }

  getAudio(id: number): Observable<Blob> {
    return this.http.get(`/api/medias/${id}/audio`, { responseType: 'blob' });
  }

  getVideo(id: number): Observable<Blob> {
    return this.http.get(`/api/medias/${id}/video`, { responseType: 'blob' });
  }

  getImage(id: number): Observable<Blob> {
    return this.http.get(`/api/medias/${id}/image`, { responseType: 'blob' });
  }

  getText(id: number): Observable<TextResponse> {
    return this.http.get<TextResponse>(`/api/medias/${id}/text`);
  }

  getMedia(id: number): Observable<Blob> {
    return this.http.get(`/api/medias/${id}/media`, { responseType: 'blob' });
  }

  vote(id: number, label: 'good' | 'bad'): Observable<VoteResponse> {
    return this.http.post<VoteResponse>(`/api/medias/${id}/vote`, { vote: label });
  }

  addToPile(file: File, label: 'good' | 'bad'): Observable<{ ok: boolean; media_id: number; is_new: boolean }> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('label', label);
    return this.http.post<{ ok: boolean; media_id: number; is_new: boolean }>('/api/medias/add-to-pile', formData);
  }
}
