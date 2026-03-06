import { Component, HostListener } from '@angular/core';
import { Router, RouterOutlet, NavigationEnd } from '@angular/router';
import { CommonModule } from '@angular/common';
import { filter } from 'rxjs/operators';
import { DialogHostComponent } from './components/dialog-host/dialog-host.component';
import { DetectorExportModalComponent } from './components/modals/detector-export-modal/detector-export-modal.component';
import { SettingsModalComponent } from './components/modals/settings-modal/settings-modal.component';
import { VtDialogService } from './services/dialog.service';

@Component({
  selector: 'app-root',
  imports: [CommonModule, RouterOutlet, DialogHostComponent, DetectorExportModalComponent, SettingsModalComponent],
  templateUrl: './app.component.html',
  styleUrl: './app.component.scss',
})
export class AppComponent {
  title = 'VTSearch';
  menuOpen = false;
  showDetectorExport = false;
  showSettings = false;
  isOnLabelView = false;
  labelsStatus = '';
  detectorStatus = '';

  constructor(
    private router: Router,
    private dialog: VtDialogService,
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

  onImportLabels(): void {
    if (!this.isOnLabelView) return;
    this.menuOpen = false;
    this.dialog.alert('Label import not yet available in the Angular frontend.', 'info');
  }

  onExportDetector(): void {
    if (!this.isOnLabelView) return;
    this.menuOpen = false;
    this.showDetectorExport = true;
  }

  onExportLabels(): void {
    if (!this.isOnLabelView) return;
    this.menuOpen = false;
    this.dialog.alert('Label export not yet available in the Angular frontend.', 'info');
  }

  onSettings(): void {
    this.menuOpen = false;
    this.showSettings = true;
  }
}
