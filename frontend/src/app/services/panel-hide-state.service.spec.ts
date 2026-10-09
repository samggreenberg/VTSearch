import { TestBed } from '@angular/core/testing';
import { HttpTestingController } from '@angular/common/http/testing';

import { PanelHideStateService } from './panel-hide-state.service';
import { SettingsStateService } from './settings-state.service';
import { provideHttpTesting } from '../testing/test-providers';
import { settleResource } from '../testing/settle-resource';

describe('PanelHideStateService (#4673)', () => {
  let service: PanelHideStateService;
  let httpMock: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({ providers: [...provideHttpTesting()] });
    service = TestBed.inject(PanelHideStateService);
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => httpMock.verify());

  async function loadSettings(body: Record<string, unknown>): Promise<void> {
    TestBed.inject(SettingsStateService).load();
    TestBed.tick();
    await settleResource();
    httpMock.expectOne('/api/settings').flush(body);
    await settleResource();
  }

  it('reads both sides hidden before settings land, the default', () => {
    expect(service.left()).toBe(true);
    expect(service.right()).toBe(true);
  });

  it('follows each setting on its own', async () => {
    await loadSettings({ hide_left_panel: true, hide_right_panel: false });
    expect(service.left()).toBe(true);
    expect(service.right()).toBe(false);
  });

  it('flips a side at once and holds it until the server answers', async () => {
    await loadSettings({ hide_left_panel: true, hide_right_panel: true });

    service.toggle('right');
    expect(service.right()).toBe(false);
    expect(service.left()).toBe(true);

    const req = httpMock.expectOne('/api/settings');
    expect(req.request.method).toBe('PUT');
    expect(req.request.body).toEqual({ hide_right_panel: false });
    req.flush({ hide_left_panel: true, hide_right_panel: false });
    expect(service.right()).toBe(false);
  });

  it('keeps the newest click when an older write answers after it', async () => {
    await loadSettings({ hide_left_panel: true, hide_right_panel: true });

    service.set('left', false);
    service.set('left', true);
    const [first, second] = httpMock.match('/api/settings');

    // The first write's echo says "open"; the newer click said "hidden".
    first.flush({ hide_left_panel: false, hide_right_panel: true });
    expect(service.left()).toBe(true);

    second.flush({ hide_left_panel: true, hide_right_panel: true });
    expect(service.left()).toBe(true);
  });

  it('falls back to the saved value when the write fails', async () => {
    await loadSettings({ hide_left_panel: true, hide_right_panel: true });

    service.set('right', false);
    expect(service.right()).toBe(false);
    httpMock.expectOne('/api/settings').flush('boom', { status: 500, statusText: 'Server Error' });
    expect(service.right()).toBe(true);
  });
});
