import { Injectable, computed, inject } from '@angular/core';
import { toSignal } from '@angular/core/rxjs-interop';

import { ActiveDetectorService } from './active-detector.service';
import { AuthService } from './auth.service';
import type { ImporterField } from '../models/api.models';
import {
  resolveTemplateVars,
  type TemplateVarContext,
} from '../utils/plugin-template-vars';

/**
 * Resolves a plugin field's declared `{template_vars}` for *display*, using the
 * app's live view of who and what the server would resolve them against.
 *
 * The substitution itself is the server's job and stays there; this is the
 * preview half, so a field whose default is `"{detector_name}"` opens showing
 * the detector's name instead of the raw placeholder (issue #3199). See
 * `utils/plugin-template-vars.ts` for the rules — in particular that it belongs
 * only in **run-now** forms, never in a persisted plugin config whose template
 * must survive to be re-resolved on each later run.
 */
@Injectable({ providedIn: 'root' })
export class PluginTemplateVarsService {
  private readonly activeDetector = inject(ActiveDetectorService);
  private readonly auth = inject(AuthService);

  private readonly authStatus = toSignal(this.auth.status$, { initialValue: null });

  /** Logged-in user, `''` until `/api/auth/status` lands. */
  readonly username = computed(() => this.authStatus()?.user ?? '');

  /**
   * The context the server would resolve against: the active detector and the
   * signed-in user. `overrides` lets a caller name the detector it is actually
   * acting on when that differs from the active one (the Dashboard exports a
   * row's detector, which need not be the active one).
   */
  context(overrides?: Partial<TemplateVarContext>): TemplateVarContext {
    return {
      detectorName: this.activeDetector.detectorName(),
      detectorId: this.activeDetector.detectorId(),
      username: this.username(),
      ...overrides,
    };
  }

  /** `field`'s declared default with its declared template vars resolved. */
  resolveDefault(field: ImporterField, overrides?: Partial<TemplateVarContext>): string {
    return this.resolve(field.default ?? '', field, overrides);
  }

  /** `value` with `field`'s declared template vars resolved. */
  resolve(
    value: string,
    field: ImporterField,
    overrides?: Partial<TemplateVarContext>,
  ): string {
    return resolveTemplateVars(value, field.template_vars, this.context(overrides), field.field_type);
  }
}
