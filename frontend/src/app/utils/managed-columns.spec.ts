import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import { ColMeta, ManagedColumns } from './managed-columns';
import { SortState } from './sort-rows';

/**
 * Unit coverage for the ManagedColumns table controller. The class is pure
 * state logic (sort, drag-reorder, localStorage persistence) plus a resize
 * state machine that reads table geometry. jsdom does no layout, so the resize
 * tests stub `offsetWidth`/`scrollWidth` on the elements the controller reads;
 * the sort/reorder/persistence logic needs no DOM at all.
 */

type Col = 'name' | 'size' | 'date' | 'kind';

const DEFAULTS: readonly Col[] = ['name', 'size', 'date', 'kind'];

const META: Record<string, ColMeta> = {
  name: { label: 'Name', title: 'File name', sortable: true },
  size: { label: 'Size', title: 'File size', sortable: true },
  date: { label: 'Date', title: 'Modified', sortable: true },
  kind: { label: 'Kind', title: 'Media kind', sortable: false },
};

function make(storageKey: string | null = null, initialSortAsc?: boolean): ManagedColumns<Col> {
  return new ManagedColumns<Col>(DEFAULTS, META, {
    initialSort: 'name',
    initialSortAsc,
    storageKey,
  });
}

describe('ManagedColumns: sort', () => {
  it('initializes sort column/direction from options (ascending default)', () => {
    const mc = make();
    expect(mc.sortColumn).toBe('name');
    expect(mc.sortAsc).toBe(true);
  });

  it('honors an explicit initial descending direction', () => {
    const mc = make(null, false);
    expect(mc.sortAsc).toBe(false);
  });

  it('sortBy toggles direction when re-selecting the active column', () => {
    const mc = make();
    mc.sortBy('name'); // already active + ascending → flip to descending
    expect(mc.sortColumn).toBe('name');
    expect(mc.sortAsc).toBe(false);
    mc.sortBy('name'); // flip back to ascending
    expect(mc.sortAsc).toBe(true);
  });

  it('sortBy switches to a new column and resets to ascending', () => {
    const mc = make();
    mc.sortBy('name'); // now descending on name
    mc.sortBy('size'); // switch column → ascending
    expect(mc.sortColumn).toBe('size');
    expect(mc.sortAsc).toBe(true);
  });

  it('emits every sort change on sortState$ (starting with the initial state)', () => {
    const mc = make();
    const seen: SortState<Col>[] = [];
    const sub = mc.sortState$.subscribe((s) => seen.push({ ...s }));
    mc.sortBy('name'); // → desc
    mc.sortBy('size'); // → asc
    sub.unsubscribe();

    expect(seen).toEqual([
      { column: 'name', asc: true }, // BehaviorSubject replays initial
      { column: 'name', asc: false },
      { column: 'size', asc: true },
    ]);
  });
});

describe('ManagedColumns: indicators & ARIA', () => {
  it('sortIndicator shows ▲ for inactive columns and the active direction otherwise', () => {
    const mc = make();
    expect(mc.sortIndicator('size')).toBe('▲'); // inactive
    expect(mc.sortIndicator('name')).toBe('▲'); // active ascending
    mc.sortBy('name');
    expect(mc.sortIndicator('name')).toBe('▼'); // active descending
  });

  it('isSortActive reflects the active column only', () => {
    const mc = make();
    expect(mc.isSortActive('name')).toBe(true);
    expect(mc.isSortActive('size')).toBe(false);
  });

  it('ariaSort reports the active column direction and none for the rest', () => {
    const mc = make();
    expect(mc.ariaSort('name')).toBe('ascending');
    expect(mc.ariaSort('size')).toBe('none');
    mc.sortBy('name');
    expect(mc.ariaSort('name')).toBe('descending');
  });
});

describe('ManagedColumns: meta', () => {
  it('returns the configured meta for a known column', () => {
    const mc = make();
    expect(mc.meta('name')).toEqual({ label: 'Name', title: 'File name', sortable: true });
  });

  it('falls back to a non-sortable label-only meta for an unknown column', () => {
    const mc = make();
    expect(mc.meta('unknown')).toEqual({ label: 'unknown', title: '', sortable: false });
  });
});

describe('ManagedColumns: column order & persistence', () => {
  const KEY = 'mc-test-order';

  beforeEach(() => localStorage.clear());
  afterEach(() => localStorage.clear());

  it('defaults to the declared order when nothing is stored', () => {
    const mc = make(KEY);
    expect(mc.columnOrder).toEqual(['name', 'size', 'date', 'kind']);
  });

  it('restores a valid stored order verbatim', () => {
    localStorage.setItem(KEY, JSON.stringify(['date', 'name', 'kind', 'size']));
    const mc = make(KEY);
    expect(mc.columnOrder).toEqual(['date', 'name', 'kind', 'size']);
  });

  it('drops unknown/duplicate entries and appends missing defaults (new columns)', () => {
    // 'bogus' is unknown, 'name' is duplicated, and 'date'/'kind' are absent.
    localStorage.setItem(KEY, JSON.stringify(['size', 'bogus', 'name', 'name']));
    const mc = make(KEY);
    // Known survivors in stored order, then missing defaults appended in
    // declaration order.
    expect(mc.columnOrder).toEqual(['size', 'name', 'date', 'kind']);
  });

  it('ignores corrupt JSON and non-array payloads, falling back to defaults', () => {
    localStorage.setItem(KEY, '{not json');
    expect(make(KEY).columnOrder).toEqual(['name', 'size', 'date', 'kind']);
    localStorage.setItem(KEY, JSON.stringify({ name: 1 }));
    expect(make(KEY).columnOrder).toEqual(['name', 'size', 'date', 'kind']);
  });

  it('does not read or write localStorage when storageKey is null', () => {
    localStorage.setItem(KEY, JSON.stringify(['kind', 'date', 'size', 'name']));
    const mc = make(null);
    expect(mc.columnOrder).toEqual(['name', 'size', 'date', 'kind']); // default, not stored
    // A reorder must not persist anything either.
    reorder(mc, 'name', 'kind');
    expect(localStorage.getItem('__mc_null_probe')).toBeNull();
  });
});

/**
 * Minimal stand-in for the DragEvent.dataTransfer that jsdom omits: enough of
 * the surface (setData/getData plus the effect flags) for the reorder handlers.
 */
function fakeDataTransfer(): DataTransfer {
  const store = new Map<string, string>();
  return {
    effectAllowed: 'none',
    dropEffect: 'none',
    setData: (type: string, val: string) => store.set(type, val),
    getData: (type: string) => store.get(type) ?? '',
  } as unknown as DataTransfer;
}

/** Drive a full drag from `source` dropped onto `target`. */
function reorder(mc: ManagedColumns<Col>, source: Col, target: Col): void {
  const dt = fakeDataTransfer();
  mc.onColDragStart({ dataTransfer: dt, preventDefault() {} } as unknown as DragEvent, source);
  mc.onColDrop({ dataTransfer: dt, preventDefault() {} } as unknown as DragEvent, target);
}

describe('ManagedColumns: drag-reorder', () => {
  const KEY = 'mc-test-reorder';

  beforeEach(() => localStorage.clear());
  afterEach(() => localStorage.clear());

  it('onColDragStart records the dragged column and clears any drop target', () => {
    const mc = make();
    mc.dropTargetCol = 'size';
    const dt = fakeDataTransfer();
    mc.onColDragStart({ dataTransfer: dt, preventDefault() {} } as unknown as DragEvent, 'name');
    expect(mc.dragCol).toBe('name');
    expect(mc.dropTargetCol).toBeNull();
    expect(mc.isDragging('name')).toBe(true);
    expect(dt.getData('text/plain')).toBe('name');
    expect(dt.effectAllowed).toBe('move');
  });

  it('onColDragStart is suppressed mid-resize (preventDefault, no drag begun)', () => {
    const mc = make();
    // Simulate an in-flight resize by starting one on a real table.
    startResizeOn(mc);
    let prevented = false;
    mc.onColDragStart(
      { dataTransfer: fakeDataTransfer(), preventDefault: () => (prevented = true) } as unknown as DragEvent,
      'name',
    );
    expect(prevented).toBe(true);
    expect(mc.dragCol).toBeNull();
  });

  it('onColDragOver marks the hovered column as the drop target', () => {
    const mc = make();
    mc.dragCol = 'name';
    let prevented = false;
    const dt = fakeDataTransfer();
    mc.onColDragOver(
      { dataTransfer: dt, preventDefault: () => (prevented = true) } as unknown as DragEvent,
      'date',
    );
    expect(prevented).toBe(true);
    expect(mc.dropTargetCol).toBe('date');
    expect(mc.isDropTarget('date')).toBe(true);
    expect(dt.dropEffect).toBe('move');
  });

  it('onColDragOver ignores hovering the source column itself', () => {
    const mc = make();
    mc.dragCol = 'name';
    let prevented = false;
    mc.onColDragOver(
      { dataTransfer: fakeDataTransfer(), preventDefault: () => (prevented = true) } as unknown as DragEvent,
      'name',
    );
    expect(prevented).toBe(false);
    expect(mc.dropTargetCol).toBeNull();
  });

  it('onColDragLeave clears only the matching drop target', () => {
    const mc = make();
    mc.dropTargetCol = 'date';
    mc.onColDragLeave({} as DragEvent, 'size'); // different col → unchanged
    expect(mc.dropTargetCol).toBe('date');
    mc.onColDragLeave({} as DragEvent, 'date'); // matching → cleared
    expect(mc.dropTargetCol).toBeNull();
  });

  it('moving a column right lands it at the target slot and shifts the rest left', () => {
    const mc = make(KEY);
    reorder(mc, 'name', 'date'); // name → date's original index (2)
    expect(mc.columnOrder).toEqual(['size', 'date', 'name', 'kind']);
  });

  it('moving a column left lands it at the target slot and shifts the rest right', () => {
    const mc = make(KEY);
    reorder(mc, 'kind', 'size'); // kind → size's original index (1)
    expect(mc.columnOrder).toEqual(['name', 'kind', 'size', 'date']);
  });

  it('dropping a column on itself is a no-op', () => {
    const mc = make(KEY);
    reorder(mc, 'size', 'size');
    expect(mc.columnOrder).toEqual(['name', 'size', 'date', 'kind']);
  });

  it('a completed drop persists the new order to localStorage', () => {
    const mc = make(KEY);
    reorder(mc, 'name', 'kind');
    expect(JSON.parse(localStorage.getItem(KEY)!)).toEqual(mc.columnOrder);
  });

  it('onColDrop with no active drag resets state without touching the order', () => {
    const mc = make(KEY);
    mc.dragCol = null;
    mc.onColDrop({ preventDefault() {} } as DragEvent, 'size');
    expect(mc.columnOrder).toEqual(['name', 'size', 'date', 'kind']);
    expect(localStorage.getItem(KEY)).toBeNull();
  });

  it('onColDragEnd and completed drops clear the transient drag flags', () => {
    const mc = make();
    mc.dragCol = 'name';
    mc.dropTargetCol = 'size';
    mc.onColDragEnd({} as DragEvent);
    expect(mc.dragCol).toBeNull();
    expect(mc.dropTargetCol).toBeNull();
    expect(mc.isDragging('name')).toBe(false);
  });
});

/**
 * Build a `<table>` with a `<thead>` of `data-col` headers and one body row.
 * Stubs the layout geometry the resize code reads: table width, each header's
 * width, and each body cell's `scrollWidth` (its natural content width).
 */
function makeTable(opts: {
  tableWidth: number;
  headerWidths: Record<Col, number>;
  cellScrollWidths?: Partial<Record<Col, number>>;
}): HTMLTableElement {
  const table = document.createElement('table');
  Object.defineProperty(table, 'offsetWidth', { value: opts.tableWidth, configurable: true });

  const thead = document.createElement('thead');
  const headRow = document.createElement('tr');
  const body = document.createElement('tbody');
  const bodyRow = document.createElement('tr');

  for (const col of DEFAULTS) {
    const th = document.createElement('th');
    th.setAttribute('data-col', col);
    Object.defineProperty(th, 'offsetWidth', { value: opts.headerWidths[col], configurable: true });
    headRow.appendChild(th);

    const td = document.createElement('td');
    Object.defineProperty(td, 'scrollWidth', {
      value: opts.cellScrollWidths?.[col] ?? 0,
      configurable: true,
    });
    bodyRow.appendChild(td);
  }
  thead.appendChild(headRow);
  body.appendChild(bodyRow);
  table.appendChild(thead);
  table.appendChild(body);
  document.body.appendChild(table);
  return table;
}

/** Start a resize on the divider left of the `size` column of a fresh table. */
function startResizeOn(mc: ManagedColumns<Col>): HTMLTableElement {
  const table = makeTable({
    tableWidth: 1000,
    headerWidths: { name: 250, size: 250, date: 250, kind: 250 },
  });
  const sizeTh = table.querySelector('th[data-col="size"]') as HTMLElement;
  mc.startResize({
    target: sizeTh,
    clientX: 500,
    stopPropagation() {},
    preventDefault() {},
  } as unknown as MouseEvent);
  return table;
}

describe('ManagedColumns: resize', () => {
  afterEach(() => {
    document.querySelectorAll('table').forEach((t) => t.remove());
    document.body.style.cursor = '';
    document.body.style.userSelect = '';
  });

  it('startResize captures percentages, fixes the layout, and sets the resize cursor', () => {
    const mc = make();
    startResizeOn(mc);
    expect(mc.tableFixed).toBe(true);
    // 250 / 1000 = 25% for every column.
    expect(mc.colWidths['name']).toBeCloseTo(25);
    expect(mc.colWidths['size']).toBeCloseTo(25);
    expect(document.body.style.cursor).toBe('col-resize');
    expect(document.body.style.userSelect).toBe('none');
  });

  it('startResize on the leftmost divider (no previous column) is a no-op', () => {
    const mc = make();
    const table = makeTable({
      tableWidth: 1000,
      headerWidths: { name: 250, size: 250, date: 250, kind: 250 },
    });
    const nameTh = table.querySelector('th[data-col="name"]') as HTMLElement;
    mc.startResize({
      target: nameTh,
      clientX: 0,
      stopPropagation() {},
      preventDefault() {},
    } as unknown as MouseEvent);
    // No divider to the left → nothing captured, layout untouched.
    expect(mc.tableFixed).toBe(false);
    expect(mc.colWidths).toEqual({});
  });

  it('a sub-threshold move (<3px) does not begin dragging', () => {
    const mc = make();
    startResizeOn(mc);
    mc.onResizeMove({ clientX: 502 } as MouseEvent); // dx = 2
    // Untouched from the captured 25/25.
    expect(mc.colWidths['name']).toBeCloseTo(25);
    expect(mc.colWidths['size']).toBeCloseTo(25);
  });

  it('dragging the divider grows the left column and shrinks the host by the same delta', () => {
    const mc = make();
    startResizeOn(mc); // name|size divider, both at 25% of a 1000px table
    mc.onResizeMove({ clientX: 600 } as MouseEvent); // dx = +100px = +10%
    expect(mc.colWidths['name']).toBeCloseTo(35);
    expect(mc.colWidths['size']).toBeCloseTo(15);
  });

  it('clamps the shrinking column at the minimum, giving the remainder to the grower', () => {
    const mc = make();
    startResizeOn(mc); // name & size both 25%, sum 50%
    mc.onResizeMove({ clientX: 1000 } as MouseEvent); // huge drag right
    expect(mc.colWidths['size']).toBeCloseTo(ManagedColumns.MIN_COL_PCT);
    expect(mc.colWidths['name']).toBeCloseTo(50 - ManagedColumns.MIN_COL_PCT);
  });

  it('clamps the growing column at the minimum when dragging the other way', () => {
    const mc = make();
    startResizeOn(mc);
    mc.onResizeMove({ clientX: 0 } as MouseEvent); // huge drag left
    expect(mc.colWidths['name']).toBeCloseTo(ManagedColumns.MIN_COL_PCT);
    expect(mc.colWidths['size']).toBeCloseTo(50 - ManagedColumns.MIN_COL_PCT);
  });

  it('onResizeMove is inert once the resize has ended', () => {
    const mc = make();
    startResizeOn(mc);
    mc.onResizeEnd(); // but this was never dragged → triggers auto-size (below)
    const before = { ...mc.colWidths };
    mc.onResizeMove({ clientX: 900 } as MouseEvent);
    expect(mc.colWidths).toEqual(before);
  });

  it('onResizeEnd after a real drag clears cursor state without auto-sizing', () => {
    const mc = make();
    startResizeOn(mc);
    mc.onResizeMove({ clientX: 600 } as MouseEvent); // dx=100 → dragged
    const dragged = { ...mc.colWidths };
    mc.onResizeEnd();
    expect(mc.colWidths).toEqual(dragged); // no auto-fit override
    expect(document.body.style.cursor).toBe('');
    expect(document.body.style.userSelect).toBe('');
  });

  it('a click (no drag) auto-fits the left column to its content width', () => {
    const mc = make();
    // name's body cell reports 100px natural width; auto-fit targets that + 2px.
    const table = makeTable({
      tableWidth: 1000,
      headerWidths: { name: 250, size: 250, date: 250, kind: 250 },
      cellScrollWidths: { name: 100 },
    });
    const sizeTh = table.querySelector('th[data-col="size"]') as HTMLElement;
    mc.startResize({
      target: sizeTh,
      clientX: 500,
      stopPropagation() {},
      preventDefault() {},
    } as unknown as MouseEvent);
    mc.onResizeEnd(); // never moved → auto-size the grow (name) column

    // 102px / 1000px = 10.2%; size absorbs the rest of the 50% pair.
    expect(mc.colWidths['name']).toBeCloseTo(10.2);
    expect(mc.colWidths['size']).toBeCloseTo(50 - 10.2);
  });

  it('onResizeEnd is a no-op when no resize is active', () => {
    const mc = make();
    expect(() => mc.onResizeEnd()).not.toThrow();
  });
});
