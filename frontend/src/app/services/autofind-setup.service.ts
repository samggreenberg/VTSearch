import { Injectable, inject } from '@angular/core';
import { DatasetStateService } from './dataset-state.service';
import { DetectorsRegistryApiService } from './detectors-registry-api.service';
import { VtDialogService } from './dialog.service';
import { SettingsPanelService } from './settings-panel.service';
import { SettingsStateService } from './settings-state.service';

type SetupChoice = 'later' | 'import' | 'setup';

/**
 * The question after a user's first "Move to AutoFind" (#5344), from the
 * Dashboard's ⋯ menu or the Tune verdict alike: where should AutoFind's
 * results go? Asked only when the moved detector is the only one on the
 * user's AutoFind tab and no AutoFind exporter is set, which is a user new to
 * AutoFind or one on a fresh server whose settings did not come along. An
 * exporter set to None on purpose leaves that unchanged, so someone who keeps
 * results in the app hears the question again only after emptying the tab.
 */
@Injectable({ providedIn: 'root' })
export class AutoFindSetupService {
  private readonly dialog = inject(VtDialogService);
  private readonly settingsPanel = inject(SettingsPanelService);
  private readonly settingsState = inject(SettingsStateService);
  private readonly datasetState = inject(DatasetStateService);
  private readonly detectorsRegistryApi = inject(DetectorsRegistryApiService);

  /** Call once *detector* is on AutoFind. Resolves when the question, and any
   *  Settings session it opened, is over. */
  async offerAfterMove(detector: { id: string; name: string }): Promise<void> {
    if (!this.isFirstWithoutExporter(detector.id)) return;
    const choice = await this.dialog.choose<SetupChoice>(
      `"${detector.name}" is your first AutoFind detector: it now runs on each dataset you import.\n` +
        'Its results stay in the app unless you choose where AutoFind sends them. ' +
        'On a new server, import your saved settings to bring your AutoFind setup back.',
      [
        { label: 'Not now', primary: false, value: 'later' },
        { label: 'Import settings…', primary: false, value: 'import' },
        { label: 'Set up AutoFind…', primary: true, value: 'setup' },
      ],
    );
    if (choice === 'setup') {
      await this.settingsPanel.open({ tab: 'autofind' });
    } else if (choice === 'import') {
      const { imported } = await this.settingsPanel.open({ tab: 'autofind', importer: true });
      if (imported) this.keepOnAutofind(detector.id);
    }
  }

  private isFirstWithoutExporter(detectorId: string): boolean {
    const settings = this.settingsState.settingsSignal();
    // Unknown until settings load: say nothing rather than guess.
    if (!settings || settings.autofind_exporter) return false;
    return !this.datasetState.detectors.some((d) => d.autofind && d.id !== detectorId);
  }

  /** An import replaces the AutoFind list with the file's, which can drop the
   *  detector moved a moment ago, so put it back. Refresh either way: the
   *  import may have changed which detectors are on the tab. */
  private keepOnAutofind(detectorId: string): void {
    this.detectorsRegistryApi.setAutofind(detectorId, true).subscribe({
      next: () => this.datasetState.refresh(),
      // The global error interceptor surfaces the failure in the banner.
      error: () => this.datasetState.refresh(),
    });
  }
}
