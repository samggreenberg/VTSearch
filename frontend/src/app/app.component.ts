import { Component, HostListener } from '@angular/core';
import { Router, RouterOutlet, NavigationEnd } from '@angular/router';
import { CommonModule } from '@angular/common';
import { combineLatest } from 'rxjs';
import { filter } from 'rxjs/operators';
import { DialogHostComponent } from './components/dialog-host/dialog-host.component';
import { ToastContainerComponent } from './components/toast-container/toast-container.component';
import { AchievementUnlockHostComponent } from './components/achievement-unlock-host/achievement-unlock-host.component';
import { SettingsModalComponent } from './components/modals/settings-modal/settings-modal.component';
import { AchievementsModalComponent } from './components/modals/achievements-modal/achievements-modal.component';
import { KeyboardHelpModalComponent } from './components/modals/keyboard-help-modal/keyboard-help-modal.component';
import { LoginComponent } from './components/login/login.component';
import { ContextPulldownComponent } from './components/context-pulldown/context-pulldown.component';
import { IncompatiblePairExplainerComponent } from './components/context-pulldown/incompatible-pair-explainer.component';
// Importer / new-detector modals are imported here AND used exclusively
// inside `@defer` blocks in the template; Angular splits them into
// lazy chunks (they drag in the file browser, crop modal, etc., which
// together push the initial bundle over budget when eagerly loaded).
import { DatasetImporterModalComponent } from './components/dashboard/dataset-importer-modal/dataset-importer-modal.component';
import { NewDetectorModalComponent } from './components/dashboard/new-detector-modal/new-detector-modal.component';
import { MediaStateService } from './services/media-state.service';
import { DatasetStateService } from './services/dataset-state.service';
import { ActiveContextService } from './services/active-context.service';
import { RecentSessionsService } from './services/recent-sessions.service';
import { AuthService } from './services/auth.service';
import { AchievementsService } from './services/achievements.service';
import { SettingsStateService } from './services/settings-state.service';
import { ThemeService } from './services/theme.service';
import { ToastService } from './services/toast.service';
import { ActiveContextWatcherService } from './services/active-context-watcher.service';
import {
  NewThingFlowsService,
  ImporterFlowState,
  NewDetectorFlowState,
} from './services/new-thing-flows.service';
import { DemoDataset } from './models/api.models';
import type { RecentSession } from './generated/api-client/models/recent-session';
import { isPairCompatible } from './utils/context-compat';

@Component({
  selector: 'app-root',
  imports: [
    CommonModule,
    RouterOutlet,
    DialogHostComponent,
    ToastContainerComponent,
    AchievementUnlockHostComponent,
    SettingsModalComponent,
    AchievementsModalComponent,
    KeyboardHelpModalComponent,
    LoginComponent,
    ContextPulldownComponent,
    IncompatiblePairExplainerComponent,
    DatasetImporterModalComponent,
    NewDetectorModalComponent,
  ],
  templateUrl: './app.component.html',
  styleUrl: './app.component.scss',
})
export class AppComponent {
  title = 'VTSearch';
  menuOpen = false;
  showSettings = false;
  showAchievements = false;
  achievementsDisabled = false;
  showKeyboardHelp = false;
  gearClosing = false;
  isOnLabelView = false;
  settingsViewTab = '';
  /** True when the current route consumes the active pair (label / find)
   *  and the pair is not compatible, so the explainer takes over the
   *  router-outlet area in that state. */
  showIncompatibleExplainer = false;

  importerFlow: ImporterFlowState = {
    open: false,
    initialTab: '',
    guessedMediaType: '',
    guessedMediaEmbedder: '',
  };
  newDetectorFlow: NewDetectorFlowState = {
    open: false,
    defaultMediaType: '',
  };
  recentSessions: RecentSession[] = [];

  constructor(
    private router: Router,
    private mediaState: MediaStateService,
    private datasetState: DatasetStateService,
    private activeContext: ActiveContextService,
    private recent: RecentSessionsService,
    public auth: AuthService,
    private achievements: AchievementsService,
    private settingsState: SettingsStateService,
    private themeService: ThemeService,
    private newThingFlows: NewThingFlowsService,
    _toast: ToastService,
    activeContextWatcher: ActiveContextWatcherService,
  ) {
    this.auth.checkStatus();
    this.settingsState.load();
    this.settingsState.settings$.subscribe((s) => {
      this.achievementsDisabled = !!s?.disable_achievements;
    });
    this.achievements.refresh();
    this.themeService.loadFromSettings();
    activeContextWatcher.start();
    this.recent.refresh().subscribe();
    this.recent.sessions$.subscribe((sessions) => {
      this.recentSessions = sessions;
    });
    this.router.events
      .pipe(filter((e): e is NavigationEnd => e instanceof NavigationEnd))
      .subscribe((e) => {
        this.isOnLabelView =
          e.urlAfterRedirects.startsWith('/label') || e.urlAfterRedirects.startsWith('/find');
        this.recomputeExplainer();
      });

    combineLatest([
      this.activeContext.pair$,
      this.datasetState.datasets$,
      this.datasetState.detectors$,
    ]).subscribe(() => this.recomputeExplainer());

    this.newThingFlows.importer$.subscribe((state) => {
      this.importerFlow = state;
    });
    this.newThingFlows.newDetector$.subscribe((state) => {
      this.newDetectorFlow = state;
    });
  }

  private recomputeExplainer(): void {
    if (!this.isOnLabelView) {
      this.showIncompatibleExplainer = false;
      return;
    }
    const ds = this.activeContext.datasetId
      ? this.datasetState.datasets.find((d) => d.id === this.activeContext.datasetId) ?? null
      : null;
    const det = this.activeContext.modelId
      ? this.datasetState.detectors.find((d) => d.id === this.activeContext.modelId) ?? null
      : null;
    this.showIncompatibleExplainer = !isPairCompatible(ds, det);
  }

  toggleMenu(event: Event): void {
    event.stopPropagation();
    this.menuOpen = !this.menuOpen;
    if (this.menuOpen) {
      this.recent.refresh().subscribe();
    }
  }

  @HostListener('document:click', ['$event'])
  onDocumentClick(event: Event): void {
    if (this.menuOpen) {
      this.menuOpen = false;
    }
  }

  @HostListener('document:keydown', ['$event'])
  onDocumentKeydown(event: KeyboardEvent): void {
    if (event.key !== '?') return;
    if (event.ctrlKey || event.metaKey || event.altKey) return;
    if (this.isTypingTarget(event.target)) return;
    event.preventDefault();
    this.showKeyboardHelp = !this.showKeyboardHelp;
  }

  private isTypingTarget(target: EventTarget | null): boolean {
    const el = target as HTMLElement | null;
    if (!el) return false;
    const tag = el.tagName;
    if (tag === 'INPUT') {
      const type = (el as HTMLInputElement).type;
      if (type !== 'checkbox' && type !== 'radio' && type !== 'range') return true;
    }
    if (tag === 'TEXTAREA' || tag === 'SELECT') return true;
    if (el.isContentEditable) return true;
    return false;
  }

  onKeyboardHelpClosed(): void {
    this.showKeyboardHelp = false;
  }

  onHelp(): void {
    this.menuOpen = false;
    this.showKeyboardHelp = true;
  }

  onMenuKeydown(event: KeyboardEvent): void {
    const menu = event.currentTarget as HTMLElement;
    const items = Array.from(menu.querySelectorAll('[role="menuitem"]')) as HTMLElement[];
    const currentIndex = items.indexOf(document.activeElement as HTMLElement);

    switch (event.key) {
      case 'ArrowDown':
        event.preventDefault();
        if (currentIndex < items.length - 1) items[currentIndex + 1].focus();
        else items[0].focus();
        break;
      case 'ArrowUp':
        event.preventDefault();
        if (currentIndex > 0) items[currentIndex - 1].focus();
        else items[items.length - 1].focus();
        break;
      case 'Escape':
        event.preventDefault();
        this.menuOpen = false;
        break;
      case 'Enter':
      case ' ':
        if ((document.activeElement as HTMLElement)?.getAttribute('role') === 'menuitem') {
          event.preventDefault();
          (document.activeElement as HTMLElement).click();
        }
        break;
      case 'Home':
        event.preventDefault();
        items[0]?.focus();
        break;
      case 'End':
        event.preventDefault();
        items[items.length - 1]?.focus();
        break;
    }
  }

  onDashboard(): void {
    if (!this.isOnLabelView) return;
    this.menuOpen = false;
    this.router.navigate(['/dashboard']);
  }

  onRecentClicked(session: RecentSession): void {
    this.menuOpen = false;
    this.router.navigate(['/label', session.dataset_id, session.detector_id]);
  }

  formatRecentTime(epochSec: number): string {
    if (!epochSec) return '';
    const diffSec = Math.max(0, Date.now() / 1000 - epochSec);
    if (diffSec < 60) return 'just now';
    if (diffSec < 3600) return `${Math.floor(diffSec / 60)}m ago`;
    if (diffSec < 86400) return `${Math.floor(diffSec / 3600)}h ago`;
    if (diffSec < 604800) return `${Math.floor(diffSec / 86400)}d ago`;
    return `${Math.floor(diffSec / 604800)}w ago`;
  }


  onSettings(): void {
    this.menuOpen = false;
    this.gearClosing = false;
    this.settingsViewTab = this.inferMediaType();
    this.showSettings = true;
  }

  onSettingsClosed(): void {
    this.showSettings = false;
    this.gearClosing = true;
  }

  onAchievements(): void {
    this.menuOpen = false;
    this.showAchievements = true;
  }

  onAchievementsClosed(): void {
    this.showAchievements = false;
  }

  onGearAnimationEnd(): void {
    this.gearClosing = false;
  }

  // --- Hoisted new-thing modal handlers (delegate to NewThingFlowsService) ---

  onImporterClosed(): void {
    this.newThingFlows.closeImporter();
  }

  onImportStarted(): void {
    // Emit BEFORE close so subscribers that watch open→close transitions
    // see the success event first; otherwise the close-handler runs first
    // and may clear state the success handler needs.
    this.newThingFlows.emitImportStarted();
    this.newThingFlows.closeImporter();
  }

  onDemoSelected(demo: DemoDataset): void {
    this.newThingFlows.emitDemoSelected(demo);
    this.newThingFlows.closeImporter();
  }

  onNewDetectorClosed(): void {
    this.newThingFlows.closeNewDetector();
  }

  onDetectorCreated(id: string): void {
    this.newThingFlows.emitDetectorCreated(id || '');
    this.newThingFlows.closeNewDetector();
  }

  private inferMediaType(): string {
    // From labeling view: use the media type of loaded medias
    if (this.isOnLabelView) {
      const medias = this.mediaState.medias;
      if (medias.length > 0) {
        return medias[0].media_type;
      }
      return '';
    }
    // From dashboard: if all datasets and models share a single media type, use it
    const datasets = this.datasetState.datasets;
    const models = this.datasetState.detectors;
    const types = new Set<string>();
    for (const d of datasets) {
      if (d.media_type) types.add(d.media_type);
    }
    for (const m of models) {
      if (m.media_type) types.add(m.media_type);
    }
    if (types.size === 1) {
      return [...types][0];
    }
    return '';
  }
}
