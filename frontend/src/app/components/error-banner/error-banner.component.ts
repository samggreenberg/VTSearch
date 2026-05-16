import { Component, OnDestroy, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Subscription } from 'rxjs';
import { ErrorContext, ErrorService } from '../../services/error.service';

@Component({
  selector: 'vt-error-banner',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './error-banner.component.html',
  styleUrl: './error-banner.component.scss',
})
export class ErrorBannerComponent implements OnInit, OnDestroy {
  current: ErrorContext | null = null;
  expanded = false;
  copied = false;

  private sub?: Subscription;
  private copiedTimer?: ReturnType<typeof setTimeout>;

  constructor(private errorService: ErrorService) {}

  ngOnInit(): void {
    this.sub = this.errorService.error$.subscribe((ctx) => {
      this.current = ctx;
      this.expanded = false;
      this.copied = false;
    });
  }

  ngOnDestroy(): void {
    this.sub?.unsubscribe();
    if (this.copiedTimer) clearTimeout(this.copiedTimer);
  }

  toggleDetails(): void {
    this.expanded = !this.expanded;
  }

  dismiss(): void {
    this.errorService.dismiss();
  }

  async copyDebugInfo(): Promise<void> {
    if (!this.current) return;
    const text = this.errorService.formatForClipboard(this.current);
    try {
      await navigator.clipboard.writeText(text);
      this.copied = true;
      if (this.copiedTimer) clearTimeout(this.copiedTimer);
      this.copiedTimer = setTimeout(() => {
        this.copied = false;
      }, 2000);
    } catch {
      // Clipboard API can fail in non-secure contexts or when the
      // document is not focused. Fall back to a hidden textarea trick.
      const ta = document.createElement('textarea');
      ta.value = text;
      ta.style.position = 'fixed';
      ta.style.opacity = '0';
      document.body.appendChild(ta);
      ta.select();
      try {
        document.execCommand('copy');
        this.copied = true;
        if (this.copiedTimer) clearTimeout(this.copiedTimer);
        this.copiedTimer = setTimeout(() => {
          this.copied = false;
        }, 2000);
      } catch {
        // give up silently — the user can still read the details
      }
      document.body.removeChild(ta);
    }
  }

  get extraEntries(): Array<{ key: string; value: string }> {
    const extra = this.current?.extra;
    if (!extra) return [];
    return Object.entries(extra).map(([key, value]) => ({
      key,
      value: typeof value === 'string' ? value : JSON.stringify(value),
    }));
  }
}
