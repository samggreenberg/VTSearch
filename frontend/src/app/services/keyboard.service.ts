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

/**
 * A claim on the vote keys, held by a step that votes on items of its own
 * while it is open: the balance's spot check (#4273, #4413). While a claim
 * is held, ←/→ vote and ↓/↑ navigate for its holder, even from inside a modal,
 * and nothing reaches {@link KeyboardService.action$} - the ranked list behind
 * the step never sees them.
 */
export interface KeyCapture {
  vote(direction: VoteDirection): void;
  navigate?(direction: NavDirection): void;
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

  /** Held claims on the vote keys, newest last; only the newest is served. */
  private readonly captures: KeyCapture[] = [];

  /**
   * Route the vote and navigation keys to *capture* until the returned
   * function is called.
   *
   * The ranked list's shortcuts are off while a modal is open, so a step that
   * lives in a modal would otherwise get no keys at all; and a step that did
   * not live in one would share them with the list. A claim settles both: its
   * holder gets ←/→ and ↓/↑ (without Shift), the list gets nothing, and every
   * other shortcut stays off. Typing in a field and held modifiers are
   * respected exactly as for the list, and so is OS auto-repeat: each vote is
   * a discrete press.
   */
  captureVoteKeys(capture: KeyCapture): () => void {
    this.captures.push(capture);
    return () => {
      const i = this.captures.indexOf(capture);
      if (i !== -1) this.captures.splice(i, 1);
    };
  }

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
    // A step holding the vote keys takes them ahead of the modal check below:
    // it usually is the modal.
    const capture = this.captures[this.captures.length - 1];
    if (capture) {
      this.handleCaptured(e, capture);
      return;
    }

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

  /**
   * The keys a {@link KeyCapture} holds. Focus is left where it is (unlike the
   * list's votes, which blur): the holder is a modal with a focus trap, and a
   * blur would drop focus out of it.
   */
  private handleCaptured(e: KeyboardEvent, capture: KeyCapture): void {
    if (this.isTyping()) return;
    if (e.ctrlKey || e.metaKey || e.altKey || e.shiftKey) return;
    let vote: VoteDirection | null = null;
    let nav: NavDirection | null = null;
    switch (e.key) {
      case 'ArrowRight':
        vote = 'good';
        break;
      case 'ArrowLeft':
        vote = 'bad';
        break;
      case 'ArrowUp':
        nav = 'forward';
        break;
      case 'ArrowDown':
        nav = 'back';
        break;
      default:
        return;
    }
    e.preventDefault();
    if (e.repeat) return;
    if (vote) capture.vote(vote);
    else if (nav) capture.navigate?.(nav);
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
