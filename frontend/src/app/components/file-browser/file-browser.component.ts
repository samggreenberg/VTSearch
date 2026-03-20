import { Component, EventEmitter, Input, OnInit, Output } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { FileBrowserApiService, BrowseEntry } from '../../services/file-browser-api.service';

@Component({
  selector: 'vt-file-browser',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './file-browser.component.html',
  styleUrl: './file-browser.component.scss',
})
export class FileBrowserComponent implements OnInit {
  /** Comma-separated extensions to filter files, e.g. ".csv,.json" */
  @Input() extensions = '';

  /** Current value (path) — pre-populates the filename input. */
  @Input() value = '';

  /** Placeholder text for the manual input. */
  @Input() placeholder = '';

  /** Emits the selected file path (relative to root). */
  @Output() pathSelected = new EventEmitter<string>();

  browserOpen = false;
  loading = false;
  error = '';

  directories: BrowseEntry[] = [];
  files: BrowseEntry[] = [];
  currentPath = '';
  root = '';

  /** The text value in the manual input field. */
  inputValue = '';

  constructor(private fileBrowserApi: FileBrowserApiService) {}

  ngOnInit(): void {
    this.inputValue = this.value;
  }

  /** Open the file browser panel and load the root directory. */
  openBrowser(): void {
    this.browserOpen = true;
    this.error = '';
    this.loadDirectory('');
  }

  closeBrowser(): void {
    this.browserOpen = false;
  }

  /** Navigate into a directory. */
  loadDirectory(path: string): void {
    this.loading = true;
    this.error = '';
    this.fileBrowserApi.browse(path, this.extensions).subscribe({
      next: (res) => {
        this.directories = res.directories;
        this.files = res.files;
        this.currentPath = res.current_path;
        this.root = res.root;
        this.loading = false;
      },
      error: (err) => {
        this.error = err.error?.error || 'Failed to browse directory';
        this.loading = false;
      },
    });
  }

  /** Navigate up one level. */
  goUp(): void {
    if (!this.currentPath) return;
    const parts = this.currentPath.split('/');
    parts.pop();
    this.loadDirectory(parts.join('/'));
  }

  /** Enter a subdirectory. */
  enterDirectory(dir: BrowseEntry): void {
    this.loadDirectory(dir.path);
  }

  /** Select a file and emit the full path. */
  selectFile(file: BrowseEntry): void {
    const fullPath = this.root + '/' + file.path;
    this.inputValue = fullPath;
    this.pathSelected.emit(fullPath);
    this.browserOpen = false;
  }

  /** Emit the manually typed path when the user changes it. */
  onInputChange(): void {
    this.pathSelected.emit(this.inputValue);
  }

  /** Breadcrumb segments from the current path. */
  get breadcrumbs(): string[] {
    if (!this.currentPath) return [];
    return this.currentPath.split('/');
  }

  /** Navigate to a breadcrumb index. */
  navigateBreadcrumb(index: number): void {
    const parts = this.currentPath.split('/');
    this.loadDirectory(parts.slice(0, index + 1).join('/'));
  }

  /** Format bytes as a human-readable size. */
  formatSize(bytes?: number): string {
    if (bytes === undefined || bytes === null) return '';
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  }
}
