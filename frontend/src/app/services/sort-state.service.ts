import { Injectable } from '@angular/core';
import { BehaviorSubject } from 'rxjs';

export type SortMode = 'text' | 'learned' | 'load';
export type SelectMode = 'top' | 'hard' | 'new';

export interface SortedItem {
  id: number;
  score: number;
}

@Injectable({ providedIn: 'root' })
export class SortStateService {
  private readonly sortModeSubject = new BehaviorSubject<SortMode>('text');
  private readonly selectModeSubject = new BehaviorSubject<SelectMode>('top');
  private readonly sortOrderSubject = new BehaviorSubject<SortedItem[] | null>(null);
  private readonly thresholdSubject = new BehaviorSubject<number | null>(null);
  private readonly sortBusySubject = new BehaviorSubject<boolean>(false);
  private readonly sortStatusSubject = new BehaviorSubject<string>('');
  private readonly inclusionSubject = new BehaviorSubject<number>(0);
  private readonly loadSortLabelSubject = new BehaviorSubject<string>('');

  readonly sortMode$ = this.sortModeSubject.asObservable();
  readonly selectMode$ = this.selectModeSubject.asObservable();
  readonly sortOrder$ = this.sortOrderSubject.asObservable();
  readonly threshold$ = this.thresholdSubject.asObservable();
  readonly sortBusy$ = this.sortBusySubject.asObservable();
  readonly sortStatus$ = this.sortStatusSubject.asObservable();
  readonly inclusion$ = this.inclusionSubject.asObservable();
  readonly loadSortLabel$ = this.loadSortLabelSubject.asObservable();

  get sortMode(): SortMode {
    return this.sortModeSubject.value;
  }

  get selectMode(): SelectMode {
    return this.selectModeSubject.value;
  }

  get sortOrder(): SortedItem[] | null {
    return this.sortOrderSubject.value;
  }

  get threshold(): number | null {
    return this.thresholdSubject.value;
  }

  get sortBusy(): boolean {
    return this.sortBusySubject.value;
  }

  get sortStatus(): string {
    return this.sortStatusSubject.value;
  }

  get inclusion(): number {
    return this.inclusionSubject.value;
  }

  get loadSortLabel(): string {
    return this.loadSortLabelSubject.value;
  }

  setSortMode(mode: SortMode): void {
    this.sortModeSubject.next(mode);
  }

  setSelectMode(mode: SelectMode): void {
    this.selectModeSubject.next(mode);
  }

  setSortResults(order: SortedItem[], threshold: number): void {
    this.sortOrderSubject.next(order);
    this.thresholdSubject.next(threshold);
  }

  setSortBusy(busy: boolean): void {
    this.sortBusySubject.next(busy);
  }

  setSortStatus(status: string): void {
    this.sortStatusSubject.next(status);
  }

  setInclusion(value: number): void {
    this.inclusionSubject.next(value);
  }

  setLoadSortLabel(label: string): void {
    this.loadSortLabelSubject.next(label);
  }

  clear(): void {
    this.sortModeSubject.next('text');
    this.selectModeSubject.next('top');
    this.sortOrderSubject.next(null);
    this.thresholdSubject.next(null);
    this.sortBusySubject.next(false);
    this.sortStatusSubject.next('');
    this.inclusionSubject.next(0);
    this.loadSortLabelSubject.next('');
  }
}
