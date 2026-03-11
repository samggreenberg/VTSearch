import { Component, HostListener } from '@angular/core';
import { Router, RouterOutlet, NavigationEnd } from '@angular/router';
import { CommonModule } from '@angular/common';
import { filter } from 'rxjs/operators';
import { DialogHostComponent } from './components/dialog-host/dialog-host.component';
import { SettingsModalComponent } from './components/modals/settings-modal/settings-modal.component';
import { MediaStateService } from './services/media-state.service';
import { DatasetStateService } from './services/dataset-state.service';
@Component({
  selector: 'app-root',
  imports: [CommonModule, RouterOutlet, DialogHostComponent, SettingsModalComponent],
  templateUrl: './app.component.html',
  styleUrl: './app.component.scss',
})
export class AppComponent {
  title = 'VTSearch';
  menuOpen = false;
  showSettings = false;
  isOnLabelView = false;
  settingsViewTab = '';

  constructor(
    private router: Router,
    private mediaState: MediaStateService,
    private datasetState: DatasetStateService,
  ) {
    this.router.events
      .pipe(filter((e): e is NavigationEnd => e instanceof NavigationEnd))
      .subscribe((e) => {
        this.isOnLabelView = e.urlAfterRedirects.startsWith('/label');
      });
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
    this.settingsViewTab = this.inferMediaType();
    this.showSettings = true;
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
    const models = this.datasetState.models;
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
