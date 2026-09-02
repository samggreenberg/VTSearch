import { ComponentFixture, TestBed } from '@angular/core/testing';
import { Observable, of, Subject, throwError } from 'rxjs';
import { vi } from 'vitest';

import {
  FolderBrowserComponent,
  FolderBrowserFileEntry,
  FolderBrowserListing,
} from './folder-browser.component';
import { provideZoneless } from '../../testing/zoneless-testbed';
import { settleZoneless } from '../../testing/settle-resource';

/** Build a listing, defaulting the arrays to empty. */
function listing(overrides: Partial<FolderBrowserListing> = {}): FolderBrowserListing {
  return { directories: [], files: [], ...overrides };
}

describe('FolderBrowserComponent', () => {
  let component: FolderBrowserComponent;
  let fixture: ComponentFixture<FolderBrowserComponent>;

  /** Records every path browse() is asked for. */
  let requestedPaths: string[];
  /** Per-path listings; falls back to `defaultListing`. */
  let listings: Record<string, FolderBrowserListing>;
  let defaultListing: FolderBrowserListing;

  const browseFn = (path: string): Observable<FolderBrowserListing> => {
    requestedPaths.push(path);
    return of(listings[path] ?? defaultListing);
  };

  beforeEach(async () => {
    requestedPaths = [];
    listings = {};
    defaultListing = listing();

    await TestBed.configureTestingModule({
      imports: [FolderBrowserComponent],
      providers: [...provideZoneless()],
    }).compileComponents();

    fixture = TestBed.createComponent(FolderBrowserComponent);
    component = fixture.componentInstance;
  });

  /** Set the required browse input (+ optional others) and run ngOnInit. */
  function init(inputs: Record<string, unknown> = {}): void {
    fixture.componentRef.setInput('browse', browseFn);
    for (const [k, v] of Object.entries(inputs)) {
      fixture.componentRef.setInput(k, v);
    }
    fixture.detectChanges();
  }

  /** Init at the root, then navigate into `path` the way a user would.
   *  The component always opens at the browse root, so tests that need a
   *  non-root starting directory walk there first. */
  function initAt(path: string): void {
    init();
    component.onRowDblClick({ kind: 'dir', name: path, path });
    requestedPaths = [];
  }

  const dir = (name: string, extra: Record<string, unknown> = {}) => ({
    name,
    path: name,
    ...extra,
  });
  const file = (name: string, extra: Record<string, unknown> = {}) => ({
    name,
    path: name,
    ...extra,
  });

  // ------------------------------------------------------------------
  // Loading
  // ------------------------------------------------------------------

  it('loads the initial path on init and lists dirs before files', () => {
    defaultListing = listing({
      directories: [dir('beta'), dir('alpha')],
      files: [file('z.txt'), file('a.txt')],
      currentPath: '',
      rootPath: '/srv',
    });
    init();

    expect(requestedPaths).toEqual(['']);
    // Directories come first (sorted), then files (sorted).
    expect(component.rows().map(r => r.name)).toEqual(['alpha', 'beta', 'a.txt', 'z.txt']);
    expect(component.rows().map(r => r.kind)).toEqual(['dir', 'dir', 'file', 'file']);
    expect(component.loading()).toBe(false);
    expect(component.rootPath()).toBe('/srv');
  });

  it('navigating into a directory loads and records that path', () => {
    init();
    requestedPaths = [];
    component.onRowDblClick({ kind: 'dir', name: 'rock', path: 'music/rock' });
    expect(requestedPaths).toEqual(['music/rock']);
    expect(component.currentPath()).toBe('music/rock');
  });

  it('omits files when showFiles is false', () => {
    defaultListing = listing({ directories: [dir('d')], files: [file('f.txt')] });
    init({ showFiles: false });
    expect(component.rows().map(r => r.kind)).toEqual(['dir']);
  });

  it('falls back to the requested path when the listing omits currentPath', () => {
    listings['deep/dir'] = listing({ directories: [] });
    initAt('deep/dir');
    expect(component.currentPath()).toBe('deep/dir');
  });

  it('emits pathChange with the resolved path and rootPath', () => {
    defaultListing = listing({ currentPath: 'a', rootPath: '/srv' });
    const events: { path: string; rootPath: string }[] = [];
    fixture.componentRef.setInput('browse', browseFn);
    component.pathChange.subscribe(e => events.push(e));
    fixture.detectChanges();
    expect(events).toEqual([{ path: 'a', rootPath: '/srv' }]);
  });

  it('reload() re-requests the current directory', () => {
    initAt('x');
    component.reload();
    expect(requestedPaths).toEqual(['x']);
  });

  // ------------------------------------------------------------------
  // Zoneless repaint canaries
  // ------------------------------------------------------------------
  //
  // The listings above resolve synchronously (`of(...)`), so their callbacks
  // run inside an existing CD pass and cannot detect staleness. These two
  // drive an *asynchronous* browse and assert on the rendered DOM without a
  // manual `detectChanges()` — the state must repaint on its own, since the
  // only usage-independent trigger (`pathChange`) is not bound by every parent
  // (`vt-file-browser` binds only `(confirm)`) and the error path emits
  // nothing at all.

  it('repaints the listing after an async browse resolves (zoneless canary)', async () => {
    const subject = new Subject<FolderBrowserListing>();
    fixture.componentRef.setInput('browse', () => subject.asObservable());
    await fixture.whenStable();
    expect(fixture.nativeElement.textContent).toContain('Loading…');

    subject.next(listing({ directories: [dir('alpha')], files: [file('a.txt')] }));
    await settleZoneless(fixture);

    expect(fixture.nativeElement.textContent).not.toContain('Loading…');
    const names = Array.from(
      fixture.nativeElement.querySelectorAll('.vfb-row .vfb-name') as NodeListOf<HTMLElement>,
    ).map((el) => el.textContent?.trim());
    expect(names).toEqual(['alpha', 'a.txt']);
  });

  it('repaints the inline error after an async browse fails (zoneless canary)', async () => {
    const subject = new Subject<FolderBrowserListing>();
    fixture.componentRef.setInput('browse', () => subject.asObservable());
    await fixture.whenStable();

    subject.error({ error: { message: 'boom' } });
    await settleZoneless(fixture);

    const err = fixture.nativeElement.querySelector('.vfb-error') as HTMLElement | null;
    expect(err?.textContent?.trim()).toBe('boom');
    expect(fixture.nativeElement.textContent).not.toContain('Loading…');
  });

  // ------------------------------------------------------------------
  // Errors
  // ------------------------------------------------------------------

  it('surfaces a browse error inline and clears rows', () => {
    const err = { error: { message: 'boom' } };
    fixture.componentRef.setInput('browse', () => throwError(() => err));
    fixture.detectChanges();

    expect(component.error()).toBe('boom');
    expect(component.rows()).toEqual([]);
    expect(component.loading()).toBe(false);
  });

  it('prefers error.message, then error.error, then a generic fallback', () => {
    const cases: [unknown, string][] = [
      [{ error: { message: 'from-message' } }, 'from-message'],
      [{ error: { error: 'from-error' } }, 'from-error'],
      [{ error: {} }, 'Could not browse this folder.'],
      [{ error: { message: 123 } }, 'Could not browse this folder.'],
    ];
    for (const [err, expected] of cases) {
      fixture.componentRef.setInput('browse', () => throwError(() => err));
        component.reload();
      expect(component.error()).toBe(expected);
    }
  });

  // ------------------------------------------------------------------
  // Navigation
  // ------------------------------------------------------------------

  it('derives breadcrumbs from the current path', () => {
    initAt('a/b/c');
    expect(component.breadcrumbs).toEqual(['a', 'b', 'c']);
  });

  it('has no breadcrumbs at the root', () => {
    init();
    expect(component.breadcrumbs).toEqual([]);
  });

  it('navigateBreadcrumb loads the path up to and including the clicked crumb', () => {
    initAt('a/b/c');
    component.navigateBreadcrumb(1);
    expect(requestedPaths).toEqual(['a/b']);
  });

  it('navigateRoot loads the empty path', () => {
    initAt('a/b');
    component.navigateRoot();
    expect(requestedPaths).toEqual(['']);
  });

  it('goUp pops the last path segment', () => {
    initAt('a/b/c');
    component.goUp();
    expect(requestedPaths).toEqual(['a/b']);
  });

  it('goUp is a no-op at the root', () => {
    init();
    requestedPaths = [];
    component.goUp();
    expect(requestedPaths).toEqual([]);
  });

  it('enter() navigates into a directory', () => {
    init();
    requestedPaths = [];
    component.enter({ kind: 'dir', name: 'sub', path: 'sub' });
    expect(requestedPaths).toEqual(['sub']);
  });

  it('enter() on a file emits confirm and does not navigate', () => {
    init();
    requestedPaths = [];
    let confirmed: FolderBrowserFileEntry | undefined;
    component.confirm.subscribe(f => (confirmed = f));
    component.enter({ kind: 'file', name: 'f.wav', path: 'd/f.wav', size_bytes: 10 });
    expect(requestedPaths).toEqual([]);
    expect(confirmed).toEqual({
      name: 'f.wav',
      path: 'd/f.wav',
      modified_at: undefined,
      size_bytes: 10,
    });
  });

  it('enter() is suppressed while busy', () => {
    init({ busy: true });
    requestedPaths = [];
    let confirmed = false;
    component.confirm.subscribe(() => (confirmed = true));
    component.enter({ kind: 'dir', name: 'sub', path: 'sub' });
    component.enter({ kind: 'file', name: 'f', path: 'f' });
    expect(requestedPaths).toEqual([]);
    expect(confirmed).toBe(false);
  });

  it('onRowDblClick delegates to enter', () => {
    init();
    requestedPaths = [];
    component.onRowDblClick({ kind: 'dir', name: 'sub', path: 'sub' });
    expect(requestedPaths).toEqual(['sub']);
  });

  // ------------------------------------------------------------------
  // absolutePath
  // ------------------------------------------------------------------

  it('absolutePath is empty without a rootPath', () => {
    defaultListing = listing({ currentPath: 'a' });
    initAt('a');
    expect(component.absolutePath).toBe('');
  });

  it('absolutePath is the rootPath at the browse root', () => {
    defaultListing = listing({ rootPath: '/data', currentPath: '' });
    init();
    expect(component.absolutePath).toBe('/data');
  });

  it('absolutePath joins rootPath and currentPath', () => {
    defaultListing = listing({ rootPath: '/data', currentPath: 'a/b' });
    initAt('a/b');
    expect(component.absolutePath).toBe('/data/a/b');
  });

  it('absolutePath avoids a double slash when the root is "/"', () => {
    defaultListing = listing({ rootPath: '/', currentPath: 'etc' });
    initAt('etc');
    expect(component.absolutePath).toBe('/etc');
  });

  // ------------------------------------------------------------------
  // Sorting
  // ------------------------------------------------------------------

  it('setSort toggles direction on repeat and keeps dirs above files', () => {
    defaultListing = listing({
      directories: [dir('alpha'), dir('beta')],
      files: [file('a.txt'), file('b.txt')],
    });
    init();
    expect(component.sortDir()).toBe('asc');

    component.setSort('name'); // same key -> flip to desc
    expect(component.sortDir()).toBe('desc');
    // Dirs stay grouped above files even when descending.
    expect(component.rows().map(r => r.name)).toEqual(['beta', 'alpha', 'b.txt', 'a.txt']);
  });

  it('setSort switching keys resets direction to asc', () => {
    init();
    component.setSort('name');
    expect(component.sortDir()).toBe('desc');
    component.setSort('modified');
    expect(component.sortKey()).toBe('modified');
    expect(component.sortDir()).toBe('asc');
  });

  it('sorts names numerically (natural order)', () => {
    defaultListing = listing({ directories: [dir('item10'), dir('item2'), dir('item1')] });
    init();
    expect(component.rows().map(r => r.name)).toEqual(['item1', 'item2', 'item10']);
  });

  it('sorts files by size', () => {
    defaultListing = listing({
      files: [file('big', { size_bytes: 900 }), file('small', { size_bytes: 10 })],
    });
    init();
    component.setSort('size');
    expect(component.rows().map(r => r.name)).toEqual(['small', 'big']);
  });

  it('sorts by modified date', () => {
    defaultListing = listing({
      directories: [
        dir('newer', { modified_at: '2026-02-01' }),
        dir('older', { modified_at: '2026-01-01' }),
      ],
    });
    init();
    component.setSort('modified');
    expect(component.rows().map(r => r.name)).toEqual(['older', 'newer']);
  });

  it('sortIndicator shows an arrow only for the active key', () => {
    init();
    expect(component.sortIndicator('name')).toBe('▲');
    expect(component.sortIndicator('size')).toBe('');
    component.setSort('name');
    expect(component.sortIndicator('name')).toBe('▼');
  });

  // ------------------------------------------------------------------
  // Selection
  // ------------------------------------------------------------------

  it('selectRow clamps out-of-range indices', () => {
    defaultListing = listing({ directories: [dir('a'), dir('b')] });
    init();
    component.selectRow(-1);
    expect(component.selectedIndex()).toBe(-1);
    component.selectRow(5);
    expect(component.selectedIndex()).toBe(-1);
    component.selectRow(1);
    expect(component.selectedIndex()).toBe(1);
  });

  it('resets the selection after a sort', () => {
    defaultListing = listing({ directories: [dir('a'), dir('b')] });
    init();
    component.selectRow(1);
    component.setSort('name');
    expect(component.selectedIndex()).toBe(-1);
  });

  // ------------------------------------------------------------------
  // Keyboard
  // ------------------------------------------------------------------

  function press(key: string, extra: Partial<KeyboardEvent> = {}): KeyboardEvent {
    const e = new KeyboardEvent('keydown', { key, ...extra });
    component.onKeyDown(e);
    return e;
  }

  it('ArrowDown/ArrowUp move the selection with wrap-in from -1', () => {
    defaultListing = listing({ directories: [dir('a'), dir('b'), dir('c')] });
    init();
    press('ArrowDown');
    expect(component.selectedIndex()).toBe(0);
    press('ArrowDown');
    expect(component.selectedIndex()).toBe(1);
    press('ArrowUp');
    expect(component.selectedIndex()).toBe(0);
    // From no selection, ArrowUp lands on the last row.
    component.selectedIndex.set(-1);
    press('ArrowUp');
    expect(component.selectedIndex()).toBe(2);
  });

  it('Home and End jump to the first and last rows', () => {
    defaultListing = listing({ directories: [dir('a'), dir('b'), dir('c')] });
    init();
    press('End');
    expect(component.selectedIndex()).toBe(2);
    press('Home');
    expect(component.selectedIndex()).toBe(0);
  });

  it('Enter activates the selected row', () => {
    defaultListing = listing({ directories: [dir('sub')] });
    init();
    component.selectRow(0);
    requestedPaths = [];
    press('Enter');
    expect(requestedPaths).toEqual(['sub']);
  });

  it('Enter does nothing without a selection', () => {
    defaultListing = listing({ directories: [dir('sub')] });
    init();
    requestedPaths = [];
    press('Enter');
    expect(requestedPaths).toEqual([]);
  });

  it('Backspace navigates to the parent directory', () => {
    initAt('a/b');
    press('Backspace');
    expect(requestedPaths).toEqual(['a']);
  });

  it('ignores keys while loading', () => {
    defaultListing = listing({ directories: [dir('a')] });
    init();
    component.loading.set(true);
    component.selectedIndex.set(-1);
    press('ArrowDown');
    expect(component.selectedIndex()).toBe(-1);
  });

  it('type-ahead jumps to the first row matching the typed prefix', () => {
    defaultListing = listing({ directories: [dir('apple'), dir('banana'), dir('cherry')] });
    init();
    press('b');
    expect(component.rows()[component.selectedIndex()].name).toBe('banana');
  });

  it('type-ahead cycles through rows sharing a prefix once the buffer resets', () => {
    // The buffer accumulates within TYPEAHEAD_RESET_MS, so cycling to the next
    // same-prefix match requires letting the reset timer fire between presses.
    vi.useFakeTimers();
    try {
      defaultListing = listing({ directories: [dir('ball'), dir('bat'), dir('bell')] });
      init();
      press('b');
      const first = component.selectedIndex();
      vi.advanceTimersByTime(900); // clear the type-ahead buffer
      press('b');
      const second = component.selectedIndex();
      expect(second).not.toBe(first);
      expect(component.rows()[first].name.startsWith('b')).toBe(true);
      expect(component.rows()[second].name.startsWith('b')).toBe(true);
    } finally {
      vi.useRealTimers();
    }
  });

  // ------------------------------------------------------------------
  // Helpers
  // ------------------------------------------------------------------

  it('formatSize renders B / KB / MB and blanks for missing sizes', () => {
    init();
    expect(component.formatSize(undefined)).toBe('');
    expect(component.formatSize(512)).toBe('512 B');
    expect(component.formatSize(2048)).toBe('2.0 KB');
    expect(component.formatSize(5 * 1024 * 1024)).toBe('5.0 MB');
  });

  it('trackRow keys rows by kind and path', () => {
    init();
    expect(component.trackRow(0, { kind: 'file', name: 'f', path: 'd/f' })).toBe('file:d/f');
  });

  // ------------------------------------------------------------------
  // Lifecycle
  // ------------------------------------------------------------------

  it('unsubscribes from an in-flight browse on destroy', () => {
    const subject = new Subject<FolderBrowserListing>();
    fixture.componentRef.setInput('browse', () => subject.asObservable());
    fixture.detectChanges();
    expect(subject.observed).toBe(true);
    fixture.destroy();
    expect(subject.observed).toBe(false);
  });
});
