import { ChangeDetectionStrategy, Component, inject, input, output, signal } from '@angular/core';
import { ModalComponent } from '../../modal/modal.component';
import { DetectorsCrudApiService } from '../../../services/detectors-crud-api.service';

/**
 * Exports a saved detector as a portable, standalone scoring bundle
 * (`detector.onnx` + `manifest.json` + `README.md`). This is the deliberate
 * exception to VTSearch's "models stay in memory" rule, so the modal leads with
 * a proportional warning: the bundle persists the trained classifier (not raw
 * media or embeddings) for sharing with other parties.
 */
@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-detector-portable-export-modal',
  standalone: true,
  imports: [ModalComponent],
  templateUrl: './detector-portable-export-modal.component.html',
  styleUrl: './detector-portable-export-modal.component.scss',
})
export class DetectorPortableExportModalComponent {
  readonly detectorId = input('');
  readonly detectorName = input('');
  readonly closed = output<void>();

  private readonly api = inject(DetectorsCrudApiService);

  readonly downloading = signal(false);
  readonly error = signal('');
  readonly done = signal(false);

  close(): void {
    this.closed.emit();
  }

  download(): void {
    if (this.downloading()) return;
    this.downloading.set(true);
    this.error.set('');
    this.done.set(false);
    this.api.exportPortableBundle(this.detectorId()).subscribe({
      next: (blob) => {
        this.saveBlob(blob);
        this.downloading.set(false);
        this.done.set(true);
      },
      error: (err: unknown) => {
        this.downloading.set(false);
        void this.readError(err).then((msg) => this.error.set(msg));
      },
    });
  }

  private saveBlob(blob: Blob): void {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${this.slug(this.detectorName())}-detector.zip`;
    a.click();
    URL.revokeObjectURL(url);
  }

  /** Mirror the server's filesystem-safe slug for the download filename. */
  private slug(name: string): string {
    return (
      (name || 'detector')
        .toLowerCase()
        .replace(/[^a-z0-9_-]+/g, '_')
        .replace(/^_+|_+$/g, '') || 'detector'
    );
  }

  /**
   * The error body is a Blob (the request used `responseType: 'blob'`), so the
   * JSON error envelope arrives as binary. Read it back out for a useful
   * message; fall back to a hint about the loaded-dataset requirement.
   */
  private async readError(err: unknown): Promise<string> {
    const fallback = 'Export failed. Make sure a compatible dataset is loaded, then try again.';
    const body = (err as { error?: unknown })?.error;
    if (body instanceof Blob) {
      try {
        const parsed = JSON.parse(await body.text()) as { message?: string; error?: string };
        return parsed.message || parsed.error || fallback;
      } catch {
        return fallback;
      }
    }
    return fallback;
  }
}
