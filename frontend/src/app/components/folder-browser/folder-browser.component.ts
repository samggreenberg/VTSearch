import { CommonModule } from '@angular/common';
import {
  AfterViewInit,
  Component,
  ElementRef,
  EventEmitter,
  HostListener,
  Input,
  OnChanges,
  OnDestroy,
  OnInit,
  Output,
  SimpleChanges,
  ViewChild,
} from '@angular/core';
import { Observable, Subscription } from 'rxjs';

import { IconComponent } from '../icon/icon.component';

export interface FolderBrowserDirEntry {
  name: string;
  path: string;
  modified_at?: string;
}

export interface FolderBrowserFileEntry {
  name: string;
  path: string;
  modified_at?: string;
  size_bytes?: number;
}

/** Response shape from the caller-supplied browse function.
 *
 *  ``rootPath`` is optional because two endpoints back this UI:
 *  ``/api/browse`` omits it (intentional path leak guard), while
 *  ``/api/browse-media-files`` returns it. When present and
 *  ``showPathFooter`` is true the resolved absolute path is shown
 *  under the list. */
export interface FolderBrowserListing {
  directories: FolderBrowserDirEntry[];
  files: FolderBrowserFileEntry[];
  rootPath?: string;
  currentPath?: string;
}

export type FolderBrowserBrowseFn = (path: string) => Observable<FolderBrowserListing>;

interface Row {
  kind: 'dir' | 'file';
  name: string;
  path: string;
  modified_at?: string;
  size_bytes?: number;
}

type SortKey = 'name' | 'modified' | 'size';
type SortDir = 'asc' | 'desc';

const TYPEAHEAD_RESET_MS = 800;

/** OS-style folder/file picker panel.
 *
 *  Renders a breadcrumb on top and a list of directories + files in the
 *  middle.  Single-click highlights a row, double-click or Enter enters
 *  a directory or confirms a file.  ↑/↓ moves the selection, Backspace
 *  (or Alt+↑) goes to the parent directory.  Typing letters jumps to
 *  the first row whose name starts with the typed prefix.
 *
 *  Callers supply a ``browse`` function so the same component works on
 *  top of multiple backend endpoints (``/api/browse``,
 *  ``/api/browse-media-files``, ...).  The component never imports an
 *  API service directly. */
@Component({
  selector: 'vt-folder-browser',
  standalone: true,
  imports: [CommonModule, IconComponent],
  templateUrl: './folder-browser.component.html',
  styleUrl: './folder-browser.component.scss',
})
export class FolderBrowserComponent implements OnInit, OnChanges, OnDestroy, AfterViewInit {
  /** Required: returns the listing for a given relative path. */
  @Input({ required: true }) browse!: FolderBrowserBrowseFn;

  /** When false, only directories are rendered (the user is picking a
   *  folder, not a file inside it). */
  @Input() showFiles = true;

  /** Initial relative path to load.  Changes to this re-load the list. */
  @Input() initialPath = '';

  /** Label for the root crumb.  Defaults to "Root". */
  @Input() rootLabel = 'Root';

  /** Show the absolute-path footer (only meaningful when the browse fn
   *  returns ``rootPath``). */
  @Input() showPathFooter = false;

  /** Show the modified-at column.  Defaults to true. */
  @Input() showDate = true;

  /** Show the size column for files.  Defaults to true. */
  @Input() showSize = true;

  /** When true, file rows show a wait cursor and ignore clicks (e.g. the
   *  caller is processing a previous confirm). */
  @Input() busy = false;

  /** Empty-state message shown when the listing is empty. */
  @Input() emptyMessage = '';

  /** Whether to auto-focus the list on init so keyboard navigation works
   *  without an extra click.  Defaults to true. */
  @Input() autoFocus = true;

  /** Fired whenever the displayed directory changes.  ``path`` is
   *  relative to the browse root; ``rootPath`` is the absolute server
   *  path if the backend exposes one (empty string otherwise). */
  @Output() pathChange = new EventEmitter<{ path: string; rootPath: string }>();

  /** Fired when the user confirms a file (Enter on selected file, or
   *  double-click on a file).  Folders are never emitted — they
   *  navigate. */
  @Output() confirm = new EventEmitter<FolderBrowserFileEntry>();

  /** Fired on browse() errors so the parent can surface them in its own
   *  way.  The component also shows an inline error message. */
  @Output() loadError = new EventEmitter<unknown>();

  rows: Row[] = [];
  currentPath = '';
  rootPath = '';
  loading = false;
  error = '';
  selectedIndex = -1;
  sortKey: SortKey = 'name';
  sortDir: SortDir = 'asc';

  private typeaheadBuf = '';
  private typeaheadTimer: ReturnType<typeof setTimeout> | null = null;
  private currentSub?: Subscription;

  @ViewChild('listEl') listEl?: ElementRef<HTMLDivElement>;

  ngOnInit(): void {
    this.loadDirectory(this.initialPath || '');
  }

  ngAfterViewInit(): void {
    if (this.autoFocus) {
      // Defer one tick so the list element is in the DOM.
      setTimeout(() => this.listEl?.nativeElement.focus(), 0);
    }
  }

  ngOnChanges(changes: SimpleChanges): void {
    if ('initialPath' in changes && !changes['initialPath'].firstChange) {
      this.loadDirectory(this.initialPath || '');
    }
  }

  ngOnDestroy(): void {
    this.currentSub?.unsubscribe();
    if (this.typeaheadTimer) clearTimeout(this.typeaheadTimer);
  }

  // ------------------------------------------------------------------
  // Loading
  // ------------------------------------------------------------------

  private loadDirectory(path: string): void {
    this.loading = true;
    this.error = '';
    this.currentSub?.unsubscribe();
    this.currentSub = this.browse(path).subscribe({
      next: (res) => {
        const dirs = res.directories || [];
        const files = this.showFiles ? res.files || [] : [];
        const rows: Row[] = [];
        for (const d of dirs) {
          rows.push({ kind: 'dir', name: d.name, path: d.path, modified_at: d.modified_at });
        }
        for (const f of files) {
          rows.push({
            kind: 'file',
            name: f.name,
            path: f.path,
            modified_at: f.modified_at,
            size_bytes: f.size_bytes,
          });
        }
        this.rows = this.sortRows(rows, this.sortKey, this.sortDir);
        this.currentPath = res.currentPath ?? path;
        this.rootPath = res.rootPath ?? '';
        this.selectedIndex = -1;
        this.loading = false;
        this.pathChange.emit({ path: this.currentPath, rootPath: this.rootPath });
      },
      error: (err) => {
        const msg = (err && err.error && err.error.message) || (err && err.error && err.error.error) || 'Could not browse this folder.';
        this.error = typeof msg === 'string' ? msg : 'Could not browse this folder.';
        this.loading = false;
        this.rows = [];
        this.loadError.emit(err);
      },
    });
  }

  /** Re-load the current directory (e.g. after the caller changed a
   *  filter that affects the listing). */
  reload(): void {
    this.loadDirectory(this.currentPath);
  }

  // ------------------------------------------------------------------
  // Navigation
  // ------------------------------------------------------------------

  get breadcrumbs(): string[] {
    return this.currentPath ? this.currentPath.split('/').filter(Boolean) : [];
  }

  navigateRoot(): void {
    this.loadDirectory('');
  }

  navigateBreadcrumb(index: number): void {
    const parts = this.currentPath.split('/').filter(Boolean);
    this.loadDirectory(parts.slice(0, index + 1).join('/'));
  }

  enter(row: Row): void {
    if (this.busy) return;
    if (row.kind === 'dir') {
      this.loadDirectory(row.path);
    } else if (row.kind === 'file') {
      this.confirm.emit({
        name: row.name,
        path: row.path,
        modified_at: row.modified_at,
        size_bytes: row.size_bytes,
      });
    }
  }

  goUp(): void {
    if (!this.currentPath) return;
    const parts = this.currentPath.split('/').filter(Boolean);
    parts.pop();
    this.loadDirectory(parts.join('/'));
  }

  get absolutePath(): string {
    if (!this.rootPath) return '';
    if (!this.currentPath) return this.rootPath;
    return this.rootPath + '/' + this.currentPath;
  }

  // ------------------------------------------------------------------
  // Selection
  // ------------------------------------------------------------------

  selectRow(index: number): void {
    if (index < 0 || index >= this.rows.length) return;
    this.selectedIndex = index;
  }

  onRowDblClick(row: Row): void {
    this.enter(row);
  }

  // ------------------------------------------------------------------
  // Sorting
  // ------------------------------------------------------------------

  setSort(key: SortKey): void {
    if (this.sortKey === key) {
      this.sortDir = this.sortDir === 'asc' ? 'desc' : 'asc';
    } else {
      this.sortKey = key;
      this.sortDir = 'asc';
    }
    this.rows = this.sortRows(this.rows, this.sortKey, this.sortDir);
    this.selectedIndex = -1;
  }

  private sortRows(rows: Row[], key: SortKey, dir: SortDir): Row[] {
    const sign = dir === 'asc' ? 1 : -1;
    // Always keep directories above files (matches OS file managers).
    const dirs = rows.filter((r) => r.kind === 'dir');
    const files = rows.filter((r) => r.kind === 'file');
    const cmp = (a: Row, b: Row): number => {
      if (key === 'name') {
        return sign * a.name.localeCompare(b.name, undefined, { numeric: true, sensitivity: 'base' });
      }
      if (key === 'modified') {
        return sign * ((a.modified_at || '').localeCompare(b.modified_at || ''));
      }
      // size — directories have no size so sort by name within dirs.
      const av = a.size_bytes ?? -1;
      const bv = b.size_bytes ?? -1;
      if (av === bv) return a.name.localeCompare(b.name);
      return sign * (av - bv);
    };
    dirs.sort(cmp);
    files.sort(cmp);
    return [...dirs, ...files];
  }

  sortIndicator(key: SortKey): string {
    if (this.sortKey !== key) return '';
    return this.sortDir === 'asc' ? '▲' : '▼';
  }

  // ------------------------------------------------------------------
  // Keyboard
  // ------------------------------------------------------------------

  @HostListener('keydown', ['$event'])
  onKeyDown(e: KeyboardEvent): void {
    if (this.loading) return;
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      this.moveSelection(1);
      return;
    }
    if (e.key === 'ArrowUp') {
      e.preventDefault();
      this.moveSelection(-1);
      return;
    }
    if (e.key === 'Home') {
      e.preventDefault();
      if (this.rows.length > 0) this.selectRow(0);
      return;
    }
    if (e.key === 'End') {
      e.preventDefault();
      if (this.rows.length > 0) this.selectRow(this.rows.length - 1);
      return;
    }
    if (e.key === 'Enter') {
      e.preventDefault();
      if (this.selectedIndex >= 0) this.enter(this.rows[this.selectedIndex]);
      return;
    }
    if (e.key === 'Backspace' || (e.key === 'ArrowUp' && e.altKey)) {
      e.preventDefault();
      this.goUp();
      return;
    }
    // Type-ahead: any single printable character extends the buffer and
    // jumps the selection to the first matching row.
    if (e.key.length === 1 && !e.ctrlKey && !e.metaKey && !e.altKey) {
      this.applyTypeahead(e.key);
      e.preventDefault();
    }
  }

  private moveSelection(delta: number): void {
    if (this.rows.length === 0) return;
    let i = this.selectedIndex + delta;
    if (this.selectedIndex < 0) i = delta > 0 ? 0 : this.rows.length - 1;
    if (i < 0) i = 0;
    if (i >= this.rows.length) i = this.rows.length - 1;
    this.selectRow(i);
    this.scrollSelectionIntoView();
  }

  private scrollSelectionIntoView(): void {
    const list = this.listEl?.nativeElement;
    if (!list) return;
    const row = list.querySelectorAll('.vfb-row')[this.selectedIndex] as HTMLElement | undefined;
    row?.scrollIntoView({ block: 'nearest' });
  }

  private applyTypeahead(ch: string): void {
    this.typeaheadBuf += ch.toLowerCase();
    if (this.typeaheadTimer) clearTimeout(this.typeaheadTimer);
    this.typeaheadTimer = setTimeout(() => {
      this.typeaheadBuf = '';
      this.typeaheadTimer = null;
    }, TYPEAHEAD_RESET_MS);
    const needle = this.typeaheadBuf;
    // Search starting from the row after the current selection so
    // repeated presses cycle through matches.
    const start = this.selectedIndex >= 0 ? this.selectedIndex : 0;
    for (let off = 0; off < this.rows.length; off++) {
      const idx = (start + off + (needle.length === 1 ? 1 : 0)) % this.rows.length;
      if (this.rows[idx].name.toLowerCase().startsWith(needle)) {
        this.selectRow(idx);
        this.scrollSelectionIntoView();
        return;
      }
    }
  }

  // ------------------------------------------------------------------
  // Helpers
  // ------------------------------------------------------------------

  formatSize(bytes?: number): string {
    if (bytes === undefined || bytes === null) return '';
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  }

  trackRow(_index: number, row: Row): string {
    return row.kind + ':' + row.path;
  }
}
