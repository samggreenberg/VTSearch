import { Injectable } from '@angular/core';
import { BehaviorSubject } from 'rxjs';

/**
 * Shared state for the top bar's Data/Model display.
 *
 * On the dashboard, the dashboard component pushes selected dataset/model
 * names here.  On label/find views, names come from the active context.
 */
@Injectable({ providedIn: 'root' })
export class TopBarStateService {
  private readonly datasetLabelSubject = new BehaviorSubject<string>('None');
  private readonly modelLabelSubject = new BehaviorSubject<string>('None');

  readonly datasetLabel$ = this.datasetLabelSubject.asObservable();
  readonly modelLabel$ = this.modelLabelSubject.asObservable();

  get datasetLabel(): string {
    return this.datasetLabelSubject.value;
  }

  get modelLabel(): string {
    return this.modelLabelSubject.value;
  }

  setDatasetLabel(label: string): void {
    this.datasetLabelSubject.next(label);
  }

  setModelLabel(label: string): void {
    this.modelLabelSubject.next(label);
  }
}
