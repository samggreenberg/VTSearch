import { Component, EventEmitter, Input, OnChanges, OnInit, Output, SimpleChanges } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { map } from 'rxjs/operators';

import { FileBrowserApiService } from '../../services/file-browser-api.service';
import {
  FolderBrowserBrowseFn,
  FolderBrowserComponent,
  FolderBrowserFileEntry,
} from '../folder-browser/folder-browser.component';

/** "Text input + Browse button" field that opens an inline OS-style
 *  folder browser panel.
 *
 *  Thin wrapper around :cmp:`FolderBrowserComponent` - the field
 *  controls open/close state and binds a ``/api/browse``-backed
 *  ``browseFn`` to the unified browser. */
@Component({
  selector: 'vt-file-browser',
  standalone: true,
  imports: [CommonModule, FormsModule, FolderBrowserComponent],
  templateUrl: './file-browser.component.html',
  styleUrl: './file-browser.component.scss',
})
export class FileBrowserComponent implements OnInit, OnChanges {
  /** Comma-separated extensions to filter files, e.g. ".csv,.json" */
  @Input() extensions = '';

  /** Current value (path) - pre-populates the filename input. */
  @Input() value = '';

  /** Placeholder text for the manual input. */
  @Input() placeholder = '';

  /** Emits the selected file path (relative to root). */
  @Output() pathSelected = new EventEmitter<string>();

  browserOpen = false;

  /** The text value in the manual input field. */
  inputValue = '';

  /** Bound to the inner ``<vt-folder-browser>``.  Arrow function so
   *  ``this`` resolves correctly when the child component invokes it. */
  readonly browseFn: FolderBrowserBrowseFn = (path: string) =>
    this.fileBrowserApi.browse(path, this.extensions).pipe(
      map((res) => ({
        directories: res.directories,
        files: res.files,
        currentPath: res.current_path,
      })),
    );

  constructor(private fileBrowserApi: FileBrowserApiService) {}

  ngOnInit(): void {
    this.inputValue = this.value;
  }

  ngOnChanges(changes: SimpleChanges): void {
    if (changes['value']) {
      this.inputValue = this.value;
    }
  }

  openBrowser(): void {
    this.browserOpen = true;
  }

  closeBrowser(): void {
    this.browserOpen = false;
  }

  onFileConfirmed(entry: FolderBrowserFileEntry): void {
    this.inputValue = entry.path;
    this.pathSelected.emit(entry.path);
    this.browserOpen = false;
  }

  onInputChange(): void {
    this.pathSelected.emit(this.inputValue);
  }
}
