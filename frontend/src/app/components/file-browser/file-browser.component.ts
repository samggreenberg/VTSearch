import { ChangeDetectionStrategy, Component, effect, inject, input, output } from '@angular/core';

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
 *  Thin wrapper around :cmp:`FolderBrowserComponent`; the field
 *  controls open/close state and binds a ``/api/browse``-backed
 *  ``browseFn`` to the unified browser. */
@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-file-browser',
  standalone: true,
  imports: [FormsModule, FolderBrowserComponent],
  templateUrl: './file-browser.component.html',
  styleUrl: './file-browser.component.scss',
})
export class FileBrowserComponent {
  private fileBrowserApi = inject(FileBrowserApiService);

  /** Comma-separated extensions to filter files, e.g. ".csv,.json" */
  readonly extensions = input('');

  /** Current value (path): pre-populates the filename input. */
  readonly value = input('');

  /** Placeholder text for the manual input. */
  readonly placeholder = input('');

  /** Emits the selected file path (relative to root). */
  readonly pathSelected = output<string>();

  browserOpen = false;

  /** The text value in the manual input field. */
  inputValue = '';

  /** Bound to the inner ``<vt-folder-browser>``.  Arrow function so
   *  ``this`` resolves correctly when the child component invokes it. */
  readonly browseFn: FolderBrowserBrowseFn = (path: string) =>
    this.fileBrowserApi.browse(path, this.extensions()).pipe(
      map((res) => ({
        directories: res.directories,
        files: res.files,
        currentPath: res.current_path,
      })),
    );

  constructor() {
    // Signal inputs don't fire `ngOnChanges`, so mirror the `value` input into
    // the editable `inputValue` field whenever the parent pushes a new value
    // (also seeds it on first render). User keystrokes change `inputValue` via
    // ngModel without touching `value`, so they never trigger this reset.
    effect(() => {
      this.inputValue = this.value();
    });
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
