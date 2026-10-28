import { TestBed } from '@angular/core/testing';

import { SettingsOpenRequest, SettingsPanelService } from './settings-panel.service';

describe('SettingsPanelService', () => {
  let service: SettingsPanelService;
  let requests: SettingsOpenRequest[];

  beforeEach(() => {
    service = TestBed.inject(SettingsPanelService);
    requests = [];
    service.requests$.subscribe((r) => requests.push(r));
  });

  it('hands each open request to the shell', () => {
    void service.open({ tab: 'autofind', importer: true });
    expect(requests).toEqual([{ tab: 'autofind', importer: true }]);
  });

  it('resolves on close with no import', async () => {
    const session = service.open({ tab: 'autofind' });
    service.closed();
    await expect(session).resolves.toEqual({ imported: false });
  });

  it('reports an import applied while it was open', async () => {
    const session = service.open({ tab: 'autofind' });
    service.noteImported();
    service.closed();
    await expect(session).resolves.toEqual({ imported: true });
  });

  it('ignores an import or a close with no session open, as from the gear', async () => {
    service.noteImported();
    service.closed();
    const session = service.open({ tab: 'autofind' });
    service.closed();
    await expect(session).resolves.toEqual({ imported: false });
  });

  it('settles a superseded session rather than leaving it waiting', async () => {
    const first = service.open({ tab: 'autofind' });
    void service.open({ tab: 'general' });
    await expect(first).resolves.toEqual({ imported: false });
  });
});
