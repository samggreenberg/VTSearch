import { TestBed } from '@angular/core/testing';
import { signal } from '@angular/core';
import { of } from 'rxjs';

import { AutoFindSetupService } from './autofind-setup.service';
import { DatasetStateService } from './dataset-state.service';
import { DetectorsRegistryApiService } from './detectors-registry-api.service';
import { VtDialogService } from './dialog.service';
import { SettingsPanelService } from './settings-panel.service';
import { SettingsStateService } from './settings-state.service';
import type { AppSettings } from '../generated/api-client/models/app-settings';
import type { DetectorRegistryEntry } from '../generated/api-client/models/detector-registry-entry';

/**
 * The question after a user's first "Move to AutoFind" (#5344): asked only
 * when the moved detector is alone on the AutoFind tab and no exporter is set,
 * and each answer leads where it says.
 */
describe('AutoFindSetupService', () => {
  let service: AutoFindSetupService;
  const settings = signal<AppSettings | null>(null);
  let detectors: DetectorRegistryEntry[];
  let dialog: { choose: ReturnType<typeof vi.fn> };
  let panel: { open: ReturnType<typeof vi.fn> };
  let registryApi: { setAutofind: ReturnType<typeof vi.fn> };
  let datasetState: { detectors: DetectorRegistryEntry[]; refresh: ReturnType<typeof vi.fn> };

  const moved = { id: 'm1', name: 'Cats' };

  beforeEach(() => {
    settings.set({ autofind_exporter: '' });
    detectors = [
      { id: 'm1', name: 'Cats', autofind: true },
      { id: 'm2', name: 'Dogs' },
    ] as DetectorRegistryEntry[];
    dialog = { choose: vi.fn().mockResolvedValue('later') };
    panel = { open: vi.fn().mockResolvedValue({ imported: false }) };
    registryApi = { setAutofind: vi.fn().mockReturnValue(of({ ok: true, autofind: true })) };
    datasetState = {
      get detectors() {
        return detectors;
      },
      refresh: vi.fn(),
    };
    TestBed.configureTestingModule({
      providers: [
        { provide: SettingsStateService, useValue: { settingsSignal: settings } },
        { provide: DatasetStateService, useValue: datasetState },
        { provide: DetectorsRegistryApiService, useValue: registryApi },
        { provide: VtDialogService, useValue: dialog },
        { provide: SettingsPanelService, useValue: panel },
      ],
    });
    service = TestBed.inject(AutoFindSetupService);
  });

  it('asks on the first AutoFind detector when no exporter is set', async () => {
    await service.offerAfterMove(moved);
    expect(dialog.choose).toHaveBeenCalledTimes(1);
    const [message, buttons] = dialog.choose.mock.calls[0];
    expect(message).toContain('"Cats" is your first AutoFind detector');
    expect(buttons.map((b: { label: string }) => b.label)).toEqual([
      'Not now',
      'Import settings…',
      'Set up AutoFind…',
    ]);
    expect(panel.open).not.toHaveBeenCalled();
  });

  it('asks whether or not the registry has caught up with the move yet', async () => {
    detectors = [{ id: 'm1', name: 'Cats' }] as DetectorRegistryEntry[];
    await service.offerAfterMove(moved);
    expect(dialog.choose).toHaveBeenCalledTimes(1);
  });

  it('stays quiet when another detector is already on AutoFind', async () => {
    detectors = [...detectors, { id: 'm3', name: 'Birds', autofind: true } as DetectorRegistryEntry];
    await service.offerAfterMove(moved);
    expect(dialog.choose).not.toHaveBeenCalled();
  });

  it('stays quiet when an AutoFind exporter is set', async () => {
    settings.set({ autofind_exporter: 'email' });
    await service.offerAfterMove(moved);
    expect(dialog.choose).not.toHaveBeenCalled();
  });

  it('stays quiet until settings have loaded', async () => {
    settings.set(null);
    await service.offerAfterMove(moved);
    expect(dialog.choose).not.toHaveBeenCalled();
  });

  it('opens Settings on the AutoFind tab for Set up AutoFind', async () => {
    dialog.choose.mockResolvedValue('setup');
    await service.offerAfterMove(moved);
    expect(panel.open).toHaveBeenCalledWith({ tab: 'autofind' });
    expect(registryApi.setAutofind).not.toHaveBeenCalled();
  });

  it('opens the settings importer over the AutoFind tab for Import settings', async () => {
    dialog.choose.mockResolvedValue('import');
    await service.offerAfterMove(moved);
    expect(panel.open).toHaveBeenCalledWith({ tab: 'autofind', importer: true });
    // Nothing was imported, so nothing to put back.
    expect(registryApi.setAutofind).not.toHaveBeenCalled();
  });

  it('puts the moved detector back on AutoFind after an import replaced the list', async () => {
    dialog.choose.mockResolvedValue('import');
    panel.open.mockResolvedValue({ imported: true });
    await service.offerAfterMove(moved);
    expect(registryApi.setAutofind).toHaveBeenCalledWith('m1', true);
    expect(datasetState.refresh).toHaveBeenCalled();
  });
});
