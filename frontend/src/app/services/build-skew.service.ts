import { Injectable, computed, inject, signal } from '@angular/core';
import { SettingsApiService } from './settings-api.service';
import { ToastService } from './toast.service';
// Written at build time by frontend/scripts/build-stamp.mjs (gitignored,
// alongside the generated API client); the prebuild/pretest hooks make it.
import { BUILD_STAMP } from '../generated/build-stamp';

/** The value either side reports when it cannot determine its own commit. */
const UNKNOWN = '0.0.0-unknown';

/**
 * Warn when the SPA in the browser was built from a different commit than the
 * server it is talking to.
 *
 * `static/` is a gitignored build artifact, so `git pull && python app.py`
 * upgrades the server while leaving the bundle untouched. Nothing used to
 * report that: the Settings dialog's version line shows the *server's*
 * `vtsearch.__version__`, which reads perfectly current while the browser runs
 * months-old JavaScript. Issue #2898 is what that costs — three rounds of
 * "still doesn't work" on an exporter capability that was working in both the
 * server and the current bundle, because the reporter's bundle predated it.
 *
 * The check is deliberately blunt: any disagreement is worth a word, because
 * the bundle and the server are built from the same tree and are only ever
 * meant to move together. Neither side can self-diagnose — the browser cannot
 * see `static/`'s mtime and the server cannot see what the browser parsed — so
 * comparing the two stamps is the only thing that can catch it.
 */
@Injectable({ providedIn: 'root' })
export class BuildSkewService {
  private readonly settingsApi = inject(SettingsApiService);
  private readonly toast = inject(ToastService);

  /** The commit this bundle was built from. */
  readonly bundleVersion: string = BUILD_STAMP;

  /** The server's version, once observed; empty until the check runs. */
  readonly serverVersion = signal('');

  /**
   * True when the two stamps disagree in a way worth reporting.
   *
   * `0.0.0-unknown` on either side means that side could not determine its own
   * commit (no `.git`, no baked stamp), so a comparison would produce noise
   * rather than information — a source checkout served by a Docker-built
   * package is a normal, working setup.
   */
  readonly skewed = computed(() => {
    const server = this.serverVersion();
    if (!server || !this.bundleVersion) return false;
    if (server === UNKNOWN || this.bundleVersion === UNKNOWN) return false;
    return server !== this.bundleVersion;
  });

  /**
   * Compare the bundle's stamp against `GET /api/version` and toast on
   * mismatch. Safe to call once at startup; failures are swallowed, since a
   * version check must never be the reason the app looks broken.
   */
  check(): void {
    this.settingsApi.getVersion().subscribe({
      next: ({ version }) => {
        this.serverVersion.set((version || '').trim());
        if (!this.skewed()) return;
        // `warning`, not `error`: the app is working, it is just working from
        // older code than the server — a degraded result the user is holding
        // without knowing it, which is exactly what that level is for. It also
        // stays up until dismissed, which this needs.
        this.toast.warning({
          message: 'This page is running an out-of-date build',
          detail:
            `The server is version ${this.serverVersion()}, but the JavaScript in your browser ` +
            `was built from ${this.bundleVersion}. Rebuild the frontend ` +
            `(cd frontend && npm run build:prod) and hard-reload this page. Until you do, ` +
            `anything added since your bundle was built will appear to be missing.`,
          // Not auto-dismissed: this explains away every "feature X does
          // nothing" symptom the user is about to hit, and it stays true until
          // they act on it.
          dedupKey: 'build-skew',
        });
      },
      error: () => {
        // Offline, or a server too old to have /api/version. Either way there
        // is nothing useful to say.
      },
    });
  }
}
