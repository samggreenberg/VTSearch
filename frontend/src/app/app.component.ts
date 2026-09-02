import {
  ChangeDetectionStrategy,
  Component,
  computed,
  effect,
  HostListener,
  inject,
  signal,
} from '@angular/core';
import { Router, RouterOutlet, NavigationEnd } from '@angular/router';
import { CommonModule } from '@angular/common';
import { filter } from 'rxjs/operators';
import { DialogHostComponent } from './components/dialog-host/dialog-host.component';
import { ToastContainerComponent } from './components/toast-container/toast-container.component';
import { OfflineBannerComponent } from './components/offline-banner/offline-banner.component';
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
import { ActiveDatasetService } from './services/active-dataset.service';
import { ActiveDetectorService } from './services/active-detector.service';
import { RecentSessionsService } from './services/recent-sessions.service';
import { AuthService } from './services/auth.service';
import { HuggingFaceAuthService } from './services/huggingface-auth.service';
import { AchievementsService } from './services/achievements.service';
import { SettingsStateService } from './services/settings-state.service';
import { ThemeService } from './services/theme.service';
import { ToastService } from './services/toast.service';
import { ActiveContextWatcherService } from './services/active-context-watcher.service';
import { BuildSkewService } from './services/build-skew.service';
import {
  NewThingFlowsService,
  ImporterFlowState,
  NewDetectorFlowState,
} from './services/new-thing-flows.service';
import { DemoDatasetEntry } from './generated/api-client/models/demo-dataset-entry';
import type { RecentSession } from './generated/api-client/models/recent-session';
import { isPairCompatible } from './utils/context-compat';

@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'app-root',
  imports: [
    CommonModule,
    RouterOutlet,
    DialogHostComponent,
    ToastContainerComponent,
    OfflineBannerComponent,
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
  private router = inject(Router);
  private mediaState = inject(MediaStateService);
  private datasetState = inject(DatasetStateService);
  private readonly activeDataset = inject(ActiveDatasetService);
  private readonly activeDetector = inject(ActiveDetectorService);
  private recent = inject(RecentSessionsService);
  auth = inject(AuthService);
  achievements = inject(AchievementsService);
  private settingsState = inject(SettingsStateService);
  private themeService = inject(ThemeService);
  private newThingFlows = inject(NewThingFlowsService);
  private hfAuth = inject(HuggingFaceAuthService);
  private toast = inject(ToastService);

  title = 'VTSearch';
  menuOpen = false;
  showSettings = false;
  // Written from the `achievements.openPanelRequest$` subscribe (async) as well
  // as bound click handlers, so a signal so the panel opens under zoneless.
  readonly showAchievements = signal(false);
  // Written from a settings `effect()` (Recipe F).
  readonly achievementsDisabled = signal(false);
  showKeyboardHelp = false;
  gearClosing = false;
  achievementsClosing = false;
  helpClosing = false;
  // Written from the router-events subscribe (async); template-bound.
  readonly isOnLabelView = signal(false);
  private readonly isOnBrowseView = signal(false);
  settingsViewTab = '';
  /** True when the current route consumes the active pair (label / find)
   *  and the pair is not compatible, so the explainer takes over the
   *  router-outlet area in that state.
   *
   *  A `computed` over the route signals and the two active-context
   *  services, so it has no separate recompute step to forget to run: it
   *  re-evaluates when the route changes, when the pair changes, and when
   *  the registry lands (the case that matters on a cold deep-link, where
   *  the ids resolve to entries only after the fetch returns). */
  readonly showIncompatibleExplainer = computed(() => {
    if (!this.isOnLabelView() || this.isOnBrowseView()) return false;
    return !isPairCompatible(this.activeDataset.dataset(), this.activeDetector.detector());
  });

  // Written from the `newThingFlows.importer$`/`newDetector$` subscribes (async).
  readonly importerFlow = signal<ImporterFlowState>({
    open: false,
    initialTab: '',
    guessedMediaType: '',
    guessedMediaEmbedder: '',
  });
  readonly newDetectorFlow = signal<NewDetectorFlowState>({
    open: false,
    defaultMediaType: '',
    datasetEmbedder: '',
  });
  // Written from the `recent.sessions$` subscribe (async).
  readonly recentSessions = signal<RecentSession[]>([]);

  constructor() {
    const activeContextWatcher = inject(ActiveContextWatcherService);

    // Before anything else: if this bundle predates the server, say so. Every
    // other symptom the user is about to hit ("that button does nothing") is
    // downstream of it, and nothing else in the app can tell them (#2898).
    inject(BuildSkewService).check();
    this.auth.checkStatus();
    this.hfAuth.refresh();
    this.handleHuggingFaceRedirect();
    this.settingsState.load();
    effect(() => {
      this.achievementsDisabled.set(
        this.settingsState.settingsSignal()?.enable_achievements === false,
      );
    });
    this.achievements.refresh();
    this.achievements.openPanelRequest$.subscribe(() => {
      this.showAchievements.set(true);
      this.achievements.acknowledgeAll();
    });
    this.themeService.loadFromSettings();
    activeContextWatcher.start();
    this.recent.refresh().subscribe();
    this.recent.sessions$.subscribe((sessions) => {
      this.recentSessions.set(sessions);
    });
    this.router.events
      .pipe(filter((e): e is NavigationEnd => e instanceof NavigationEnd))
      .subscribe((e) => {
        const onBrowse = e.urlAfterRedirects.startsWith('/browse');
        this.isOnBrowseView.set(onBrowse);
        this.isOnLabelView.set(
          e.urlAfterRedirects.startsWith('/label') ||
            e.urlAfterRedirects.startsWith('/find') ||
            onBrowse,
        );
      });

    this.newThingFlows.importer$.subscribe((state) => {
      this.importerFlow.set(state);
    });
    this.newThingFlows.newDetector$.subscribe((state) => {
      this.newDetectorFlow.set(state);
    });
  }

  /**
   * Surface the outcome of a "Sign in with HuggingFace" round-trip.  The OAuth
   * callback redirects the browser to ``/?hf_auth=success|error``; we toast the
   * result, refresh sign-in state, and strip the query params so a reload
   * doesn't re-toast.
   */
  private handleHuggingFaceRedirect(): void {
    const params = new URLSearchParams(window.location.search);
    const result = params.get('hf_auth');
    if (!result) return;
    if (result === 'success') {
      this.hfAuth.refresh();
      this.toast.success({ message: 'Signed in to HuggingFace.' });
    } else {
      this.toast.error({
        message: 'HuggingFace sign-in failed.',
        detail: params.get('hf_auth_reason') || undefined,
      });
    }
    params.delete('hf_auth');
    params.delete('hf_auth_reason');
    const query = params.toString();
    const newUrl = window.location.pathname + (query ? `?${query}` : '') + window.location.hash;
    window.history.replaceState({}, '', newUrl);
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
    if (this.showKeyboardHelp) {
      this.onKeyboardHelpClosed();
    } else {
      this.onHelp();
    }
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
    this.helpClosing = true;
  }

  onHelp(): void {
    this.menuOpen = false;
    this.helpClosing = false;
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
    if (!this.isOnLabelView()) return;
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
    this.achievementsClosing = false;
    this.showAchievements.set(true);
    this.achievements.acknowledgeAll();
  }

  onAchievementsClosed(): void {
    this.showAchievements.set(false);
    this.achievementsClosing = true;
  }

  onGearAnimationEnd(): void {
    this.gearClosing = false;
  }

  onTrophyAnimationEnd(): void {
    this.achievementsClosing = false;
  }

  onHelpAnimationEnd(): void {
    this.helpClosing = false;
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

  onDemoSelected(demo: DemoDatasetEntry): void {
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
    if (this.isOnLabelView()) {
      const medias = this.mediaState.mediasSignal();
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
