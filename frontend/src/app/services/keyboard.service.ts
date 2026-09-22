import { Injectable, NgZone, OnDestroy, inject } from '@angular/core';
import { Subject } from 'rxjs';

export type VoteDirection = 'good' | 'bad';
export type ZoomDirection = 'in' | 'out';
export type RotateDirection = 'left' | 'right';
/** `back` walks the trail of items already voted on; `forward` returns to the
 *  queue (the next unlabeled item the host's advance rule picks). */
export type NavDirection = 'back' | 'forward';

export interface KeyboardAction {
  type: 'vote' | 'volume' | 'playback' | 'zoom' | 'rotate' | 'undo' | 'redo' | 'navigate';
  direction?: VoteDirection;
  volumeDelta?: number;
  zoomDirection?: ZoomDirection;
  rotateDirection?: RotateDirection;
  navDirection?: NavDirection;
}

@Injectable({ providedIn: 'root' })
export class KeyboardService implements OnDestroy {
  private zone = inject(NgZone);

  // Shortcuts are dispatched as a plain Subject emit. Under zoneless CD the
  // re-entry that used to wrap every `.next()` in `zone.run(...)` is gone: the
  // CD trigger now lives in the consumer (center-panel), whose shortcut-driven
  // state is signalized so a write from this callback notifies the scheduler.
  // The keydown listener still runs in `runOutsideAngular` (a harmless no-op
  // under zoneless, and it still avoids per-keystroke churn while prod is zoned).
  readonly action$ = new Subject<KeyboardAction>();

  private listener: ((e: KeyboardEvent) => void) | null = null;

  /** Start listening for keyboard shortcuts on the document. */
  start(): void {
    if (this.listener) return;
    this.listener = (e: KeyboardEvent) => this.handleKeydown(e);
    this.zone.runOutsideAngular(() => {
      document.addEventListener('keydown', this.listener!);
    });
  }

  /** Stop listening for keyboard shortcuts. */
  stop(): void {
    if (this.listener) {
      document.removeEventListener('keydown', this.listener);
      this.listener = null;
    }
  }

  ngOnDestroy(): void {
    this.stop();
    this.action$.complete();
  }

  private handleKeydown(e: KeyboardEvent): void {
    // Skip when a modal is open
    if (document.querySelector('.modal-backdrop')) return;

    // Skip when typing in text fields
    if (this.isTyping()) return;

    // Cmd/Ctrl-Z (undo) and Cmd/Ctrl-Shift-Z (redo) are the only modifier
    // shortcuts; everything below this point requires no modifiers.
    if ((e.ctrlKey || e.metaKey) && !e.altKey && (e.key === 'z' || e.key === 'Z')) {
      e.preventDefault();
      (document.activeElement as HTMLElement)?.blur();
      const type = e.shiftKey ? 'redo' : 'undo';
      this.action$.next({ type });
      return;
    }

    // Skip when modifier keys are held
    if (e.ctrlKey || e.metaKey || e.altKey) return;

    switch (e.key) {
      case 'ArrowRight':
        e.preventDefault();
        // Ignore OS key auto-repeat: holding the key would otherwise cast a
        // vote per repeat event, rapidly tagging item after item (often the
        // wrong ones) until the user releases. Each vote must be a discrete
        // key press. Volume/zoom/rotate below intentionally allow repeat
        // (holding the key to adjust continuously).
        if (e.repeat) break;
        (document.activeElement as HTMLElement)?.blur();
        this.action$.next({ type: 'vote', direction: 'good' });
        break;
      case 'ArrowLeft':
        e.preventDefault();
        if (e.repeat) break;
        (document.activeElement as HTMLElement)?.blur();
        this.action$.next({ type: 'vote', direction: 'bad' });
        break;
      // Up/Down navigate the labelling queue (#4032): Down returns to the item
      // just voted on, Up goes back to the next unlabeled one. Volume, which
      // these keys used to carry, moves to Shift+Up / Shift+Down — the rarer
      // command gets the modifier. Like a vote, navigation is a discrete press
      // and ignores OS auto-repeat (a held Down would otherwise race backwards
      // through the whole trail); volume still repeats, since holding the key
      // to slide the level continuously is the point of it.
      case 'ArrowUp':
        e.preventDefault();
        (document.activeElement as HTMLElement)?.blur();
        if (e.shiftKey) {
          this.action$.next({ type: 'volume', volumeDelta: 0.05 });
          break;
        }
        if (e.repeat) break;
        this.action$.next({ type: 'navigate', navDirection: 'forward' });
        break;
      case 'ArrowDown':
        e.preventDefault();
        (document.activeElement as HTMLElement)?.blur();
        if (e.shiftKey) {
          this.action$.next({ type: 'volume', volumeDelta: -0.05 });
          break;
        }
        if (e.repeat) break;
        this.action$.next({ type: 'navigate', navDirection: 'back' });
        break;
      case ' ':
        e.preventDefault();
        (document.activeElement as HTMLElement)?.blur();
        this.action$.next({ type: 'playback' });
        break;
      case '+':
      case '=':
        e.preventDefault();
        (document.activeElement as HTMLElement)?.blur();
        this.action$.next({ type: 'zoom', zoomDirection: 'in' });
        break;
      case '-':
      case '_':
        e.preventDefault();
        (document.activeElement as HTMLElement)?.blur();
        this.action$.next({ type: 'zoom', zoomDirection: 'out' });
        break;
      case '[':
        e.preventDefault();
        (document.activeElement as HTMLElement)?.blur();
        this.action$.next({ type: 'rotate', rotateDirection: 'left' });
        break;
      case ']':
        e.preventDefault();
        (document.activeElement as HTMLElement)?.blur();
        this.action$.next({ type: 'rotate', rotateDirection: 'right' });
        break;
    }
  }

  private isTyping(): boolean {
    const el = document.activeElement;
    if (!el) return false;
    const tag = el.tagName;
    if (tag === 'INPUT') {
      const type = (el as HTMLInputElement).type;
      if (type !== 'checkbox' && type !== 'radio' && type !== 'range') return true;
    }
    if (tag === 'TEXTAREA' || tag === 'SELECT') return true;
    if ((el as HTMLElement).isContentEditable) return true;
    return false;
  }
}
