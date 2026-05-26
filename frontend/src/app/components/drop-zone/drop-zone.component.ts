import { Component, ElementRef, EventEmitter, Input, Output, ViewChild } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'vt-drop-zone',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './drop-zone.component.html',
  styleUrl: './drop-zone.component.scss',
})
export class DropZoneComponent {
  @Input() label = 'Drop files here, or click to browse';
  @Input() sublabel = '';
  @Input() accept = '';
  @Input() multiple = false;
  @Input() directory = false;
  @Input() disabled = false;
  @Output() filesSelected = new EventEmitter<File[]>();

  @ViewChild('fileInput') fileInput!: ElementRef<HTMLInputElement>;

  isDragging = false;
  private dragDepth = 0;

  openPicker(): void {
    if (this.disabled) return;
    this.fileInput?.nativeElement.click();
  }

  onInputChange(event: Event): void {
    const input = event.target as HTMLInputElement;
    if (!input.files || input.files.length === 0) return;
    this.filesSelected.emit(Array.from(input.files));
    input.value = '';
  }

  onDragEnter(event: DragEvent): void {
    if (!this.hasFiles(event)) return;
    event.preventDefault();
    event.stopPropagation();
    if (this.disabled) return;
    this.dragDepth++;
    this.isDragging = true;
  }

  onDragOver(event: DragEvent): void {
    if (!this.hasFiles(event)) return;
    event.preventDefault();
    event.stopPropagation();
    if (this.disabled) return;
    if (event.dataTransfer) {
      event.dataTransfer.dropEffect = 'copy';
    }
  }

  onDragLeave(event: DragEvent): void {
    event.preventDefault();
    event.stopPropagation();
    this.dragDepth = Math.max(0, this.dragDepth - 1);
    if (this.dragDepth === 0) this.isDragging = false;
  }

  async onDrop(event: DragEvent): Promise<void> {
    event.preventDefault();
    event.stopPropagation();
    this.dragDepth = 0;
    this.isDragging = false;
    if (this.disabled) return;
    const dt = event.dataTransfer;
    if (!dt) return;

    const files: File[] = [];
    const items = dt.items;
    const canUseEntries = items && items.length > 0
      && typeof (items[0] as DataTransferItem & { webkitGetAsEntry?: () => FileSystemEntry | null }).webkitGetAsEntry === 'function';
    if (canUseEntries) {
      const entries: FileSystemEntry[] = [];
      for (let i = 0; i < items.length; i++) {
        const entry = (items[i] as DataTransferItem & { webkitGetAsEntry?: () => FileSystemEntry | null }).webkitGetAsEntry?.();
        if (entry) entries.push(entry);
      }
      for (const entry of entries) {
        await this.walkEntry(entry, '', files);
      }
    } else if (dt.files) {
      for (let i = 0; i < dt.files.length; i++) {
        files.push(dt.files[i]);
      }
    }

    if (files.length === 0) return;
    // For single-file pickers, only emit the first dropped file.
    if (!this.directory && !this.multiple) {
      this.filesSelected.emit([files[0]]);
    } else {
      this.filesSelected.emit(files);
    }
  }

  // Some drag events on a page also carry text/other payloads; only react to
  // drags that actually include files so users can still e.g. drag-select text
  // over the zone without triggering the highlight.
  private hasFiles(event: DragEvent): boolean {
    const types = event.dataTransfer?.types;
    if (!types) return false;
    for (let i = 0; i < types.length; i++) {
      if (types[i] === 'Files') return true;
    }
    return false;
  }

  private async walkEntry(entry: FileSystemEntry, prefix: string, out: File[]): Promise<void> {
    if (entry.isFile) {
      const fileEntry = entry as FileSystemFileEntry;
      const file = await new Promise<File>((resolve, reject) => {
        fileEntry.file(resolve, reject);
      });
      const rel = prefix ? `${prefix}/${file.name}` : file.name;
      // webkitRelativePath is normally a getter on File.prototype; define an
      // own property to shadow it so consumers (e.g. the dataset importer's
      // recursive-folder filter and folder-name derivation) see the path from
      // the drop root.
      try {
        Object.defineProperty(file, 'webkitRelativePath', { value: rel, configurable: true });
      } catch {
        /* some browsers may refuse to override the getter; the file still uploads */
      }
      out.push(file);
    } else if (entry.isDirectory) {
      const dirEntry = entry as FileSystemDirectoryEntry;
      const reader = dirEntry.createReader();
      const subPrefix = prefix ? `${prefix}/${entry.name}` : entry.name;
      // readEntries() returns entries in batches; call until it returns an
      // empty array to drain a large directory.
      while (true) {
        const batch: FileSystemEntry[] = await new Promise((resolve, reject) => {
          reader.readEntries(resolve, reject);
        });
        if (batch.length === 0) break;
        for (const e of batch) {
          await this.walkEntry(e, subPrefix, out);
        }
      }
    }
  }
}
