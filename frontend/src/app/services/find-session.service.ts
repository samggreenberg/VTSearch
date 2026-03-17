import { Injectable } from '@angular/core';

/**
 * Lightweight session state passed from dashboard to find view.
 * Holds the model_id selected for the find-label operation.
 */
@Injectable({ providedIn: 'root' })
export class FindSessionService {
  modelId = '';
  modelName = '';
}
