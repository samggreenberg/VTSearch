import { TestBed } from '@angular/core/testing';
import { HttpTestingController } from '@angular/common/http/testing';

import { HintsService } from './hints.service';
import { SettingsStateService } from './settings-state.service';
import { provideHttpTesting } from '../testing/test-providers';
import { settleResource } from '../testing/settle-resource';

describe('HintsService', () => {
  let hints: HintsService;
  let httpMock: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({ providers: [...provideHttpTesting()] });
    hints = TestBed.inject(HintsService);
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => httpMock.verify());

  async function loadSettings(body: object): Promise<void> {
    TestBed.inject(SettingsStateService).load();
    TestBed.tick();
    httpMock.expectOne('/api/settings').flush(body);
    await settleResource();
  }

  it('shows no hint before the settings load', () => {
    expect(hints.isShown('add-dataset')).toBe(false);
  });

  it('shows every hint by default', async () => {
    await loadSettings({});
    expect(hints.isShown('add-dataset')).toBe(true);
    expect(hints.isShown('train')).toBe(true);
  });

  it('honours the per-hint list', async () => {
    await loadSettings({ hidden_hints: ['train'] });
    expect(hints.isShown('train')).toBe(false);
    expect(hints.isShown('add-dataset')).toBe(true);
  });

  it('keeps an earlier hide in flight when a second one is sent', async () => {
    await loadSettings({ hidden_hints: ['train'] });
    hints.hide('add-dataset');
    hints.hide('add-detector');
    const [first, second] = httpMock.match('/api/settings');
    expect(first.request.body).toEqual({ hidden_hints: ['train', 'add-dataset'] });
    expect(second.request.body).toEqual({ hidden_hints: ['train', 'add-dataset', 'add-detector'] });
    expect(hints.isShown('add-dataset')).toBe(false);
    expect(hints.isShown('add-detector')).toBe(false);

    first.flush({ hidden_hints: ['train', 'add-dataset'] });
    second.flush({ hidden_hints: ['train', 'add-dataset', 'add-detector'] });
    expect(hints.isShown('add-dataset')).toBe(false);
    expect(hints.isShown('add-detector')).toBe(false);
  });

  it('lets Show All in Settings bring a hidden hint back', async () => {
    await loadSettings({});
    hints.hideAll();
    httpMock.expectOne('/api/settings').flush({ hide_all_hints: true });
    expect(hints.isShown('add-dataset')).toBe(false);

    // The Settings modal writes its draft straight through SettingsStateService.
    TestBed.inject(SettingsStateService).update({ hide_all_hints: false, hidden_hints: [] }).subscribe();
    httpMock.expectOne('/api/settings').flush({ hide_all_hints: false, hidden_hints: [] });
    expect(hints.isShown('add-dataset')).toBe(true);
  });
});
