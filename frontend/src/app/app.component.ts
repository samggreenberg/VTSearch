import { Component, HostListener } from '@angular/core';
import { Router, RouterOutlet, NavigationEnd } from '@angular/router';
import { CommonModule } from '@angular/common';
import { filter } from 'rxjs/operators';
import { DialogHostComponent } from './components/dialog-host/dialog-host.component';
import { AchievementUnlockHostComponent } from './components/achievement-unlock-host/achievement-unlock-host.component';
import { SettingsModalComponent } from './components/modals/settings-modal/settings-modal.component';
import { KeyboardHelpModalComponent } from './components/modals/keyboard-help-modal/keyboard-help-modal.component';
import { LoginComponent } from './components/login/login.component';
import { MediaStateService } from './services/media-state.service';
import { DatasetStateService } from './services/dataset-state.service';
import { ActiveContextService } from './services/active-context.service';
import { TopBarStateService } from './services/top-bar-state.service';
import { AuthService } from './services/auth.service';
import { AchievementsService } from './services/achievements.service';
import { ThemeService } from './services/theme.service';
@Component({
  selector: 'app-root',
  imports: [CommonModule, RouterOutlet, DialogHostComponent, AchievementUnlockHostComponent, SettingsModalComponent, KeyboardHelpModalComponent, LoginComponent],
  templateUrl: './app.component.html',
  styleUrl: './app.component.scss',
})
export class AppComponent {
  title = 'VTSearch';
  menuOpen = false;
  showSettings = false;
  showKeyboardHelp = false;
  gearClosing = false;
  isOnLabelView = false;
  settingsViewTab = '';
  datasetDisplayName = '';
  modelDisplayName = '';

  constructor(
    private router: Router,
    private mediaState: MediaStateService,
    private datasetState: DatasetStateService,
    private activeContext: ActiveContextService,
    public topBarState: TopBarStateService,
    public auth: AuthService,
    private achievements: AchievementsService,
    private themeService: ThemeService,
  ) {
    this.auth.checkStatus();
    this.achievements.refresh();
    this.themeService.loadFromSettings();
    this.router.events
      .pipe(filter((e): e is NavigationEnd => e instanceof NavigationEnd))
      .subscribe((e) => {
        this.isOnLabelView =
          e.urlAfterRedirects.startsWith('/label') || e.urlAfterRedirects.startsWith('/find');
        this.updateDisplayNames();
      });

    this.topBarState.datasetLabel$.subscribe(() => this.updateDisplayNames());
    this.topBarState.modelLabel$.subscribe(() => this.updateDisplayNames());
    this.activeContext.datasetId$.subscribe(() => this.updateDisplayNames());
    this.activeContext.modelId$.subscribe(() => this.updateDisplayNames());
  }

  private updateDisplayNames(): void {
    if (this.isOnLabelView) {
      const dsId = this.activeContext.datasetId;
      const ds = this.datasetState.datasets.find((d) => d.id === dsId);
      this.datasetDisplayName = ds?.name || '';

      const mId = this.activeContext.modelId;
      const m = this.datasetState.detectors.find((mod) => mod.id === mId);
      this.modelDisplayName = m?.name || '';
    } else {
      this.datasetDisplayName = this.topBarState.datasetLabel;
      this.modelDisplayName = this.topBarState.modelLabel;
    }
  }

  toggleMenu(event: Event): void {
    event.stopPropagation();
    this.menuOpen = !this.menuOpen;
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

  onGearAnimationEnd(): void {
    this.gearClosing = false;
  }

  private inferMediaType(): string {
    // From labeling view: use the media type of loaded medias
    if (this.isOnLabelView) {
      const medias = this.mediaState.medias;
      if (medias.length > 0) {
        return medias[0].type;
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
