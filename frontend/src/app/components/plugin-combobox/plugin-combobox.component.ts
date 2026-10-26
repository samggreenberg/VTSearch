import {
  ChangeDetectionStrategy,
  Component,
  ElementRef,
  afterRenderEffect,
  computed,
  effect,
  input,
  output,
  signal,
  viewChild,
} from '@angular/core';

import { FieldOptions } from '../../models/api.models';
import { fieldHintPlacement } from '../field-hint-icon/field-hint-icon.component';

let nextId = 0;

/**
 * The value a typed *text* stands for: the text itself when it is already an
 * option's value (or matches nothing), else the value of the one option whose
 * label it is. A label two options share is ambiguous, so it stays free text;
 * telling those options apart is what picking from the list is for (#5265).
 */
export function comboboxValueFor(text: string, options: readonly FieldOptions[]): string {
  if (options.some((o) => o.value === text)) return text;
  const named = options.filter((o) => o.label === text);
  return named.length === 1 ? named[0].value : text;
}

/** The options to offer while the user has typed *text*: those whose label
 *  or value contains it, ignoring case, or every option for blank text. */
export function comboboxMatches(text: string, options: readonly FieldOptions[]): FieldOptions[] {
  const needle = text.trim().toLowerCase();
  if (!needle) return [...options];
  return options.filter(
    (o) => o.label.toLowerCase().includes(needle) || o.value.toLowerCase().includes(needle),
  );
}

/**
 * The text input for a ``"select"`` ``PluginField`` with ``allow_free_text``:
 * the user may pick one of the field's options or type any value.
 *
 * This replaces a native ``<datalist>``, which cannot do what an option with a
 * separate value and label needs (issue #5265). A plugin that returns
 * ``(dataset_uuid, "my-cool-dataset")`` options wants the name shown and the
 * UUID submitted, but Chrome draws a datalist option's *value* as its headline
 * with the label in small print beneath, and picking one writes the value into
 * the box. Swapping the two would show the name, and then submit it too, so two
 * datasets of the same name could no longer be told apart.
 *
 * Here the box shows the picked option's label and the form keeps its value.
 * Each row of the list leads with the label, with the value beneath it when the
 * two differ, and the value of the current pick is shown under the box, so two
 * options of the same name stay distinguishable after one is chosen. Typed text
 * is submitted as typed, except that a label exactly one option carries submits
 * that option's value (see {@link comboboxValueFor}).
 *
 * Every plugin-field form renders it in place of an open-coded input, and
 * ``tests_lib/meta/test_plugin_field_combobox.py`` holds them to that.
 */
@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-plugin-combobox',
  standalone: true,
  imports: [],
  templateUrl: './plugin-combobox.component.html',
  styleUrl: './plugin-combobox.component.scss',
})
export class PluginComboboxComponent {
  /** The field's current form value. */
  readonly value = input<unknown>('');
  /** The field's options, as ``DynamicFieldOptions.optionsFor`` returns them. */
  readonly options = input<FieldOptions[]>([]);
  /** DOM id of the text box, so the field's `<label for="…">` points at it. */
  readonly inputId = input('');
  readonly placeholder = input('');
  readonly title = input('');
  /** Accessible name, for a form that names the field without a `<label>`. */
  readonly ariaLabel = input('');
  readonly required = input(false);
  readonly disabled = input(false);

  /** Emits the value the form should store: an option's value, or typed text. */
  readonly valueChange = output<string>();

  private readonly autoId = `vt-plugin-combobox-${++nextId}`;
  readonly id = computed(() => this.inputId() || this.autoId);
  readonly listboxId = computed(() => `${this.id()}-listbox`);

  readonly valueText = computed(() => {
    const v = this.value();
    return v === null || v === undefined ? '' : String(v);
  });
  /** The option the current value is, if it is one. */
  readonly selected = computed(() => this.options().find((o) => o.value === this.valueText()));

  /** What the user has typed since the box last showed the value, or ``null``
   *  when it is showing the value. Cleared on a pick and when focus leaves. */
  private readonly draft = signal<string | null>(null);

  /** The box's text: the draft while typing, else the picked option's label,
   *  else the value as typed. */
  readonly text = computed(() => this.draft() ?? this.selected()?.label ?? this.valueText());
  /** The current pick's value, when its label hides it. */
  readonly detail = computed(() => {
    const pick = this.selected();
    return pick && pick.value !== pick.label ? pick.value : '';
  });

  readonly open = signal(false);
  /** Keyboard-highlighted row (the ``aria-activedescendant`` target), or -1. */
  readonly activeIndex = signal(-1);
  /** The rows on offer: every option until the user types, then the matches. */
  readonly shown = computed(() => comboboxMatches(this.draft() ?? '', this.options()));
  readonly expanded = computed(() => this.open() && this.shown().length > 0);
  readonly activeDescendant = computed(() =>
    this.expanded() && this.activeIndex() >= 0 ? this.optionId(this.activeIndex()) : null,
  );

  private readonly field = viewChild.required<ElementRef<HTMLInputElement>>('field');
  private readonly listbox = viewChild<ElementRef<HTMLElement>>('listbox');

  constructor() {
    // Place the list when it opens, and again when filtering changes its
    // height; then keep the highlighted row in view.
    afterRenderEffect(() => {
      this.shown();
      const list = this.listbox()?.nativeElement;
      if (list) this.place(list);
    });
    afterRenderEffect(() => {
      const index = this.activeIndex();
      const row = this.listbox()?.nativeElement.children[index] as HTMLElement | undefined;
      row?.scrollIntoView({ block: 'nearest' });
    });
    // While it is open, follow the box as whatever holds it scrolls, and as
    // the window resizes. Captured, because a dialog body's `scroll` does not
    // bubble.
    effect((onCleanup) => {
      const list = this.listbox()?.nativeElement;
      if (!list) return;
      const follow = (): void => this.place(list);
      window.addEventListener('scroll', follow, { capture: true, passive: true });
      window.addEventListener('resize', follow, { passive: true });
      onCleanup(() => {
        window.removeEventListener('scroll', follow, { capture: true });
        window.removeEventListener('resize', follow);
      });
    });
  }

  optionId(index: number): string {
    return `${this.id()}-option-${index}`;
  }

  onInput(event: Event): void {
    const text = (event.target as HTMLInputElement).value;
    this.draft.set(text);
    this.open.set(true);
    this.activeIndex.set(-1);
    this.commit(comboboxValueFor(text, this.options()));
  }

  pick(option: FieldOptions): void {
    this.draft.set(null);
    this.close();
    this.commit(option.value);
  }

  /** Open the list. From the keyboard, highlight the current pick (or the
   *  first row) so the arrow keys have somewhere to start; from a click,
   *  highlight only the current pick, so Enter still means what it did. */
  openList(fromKeyboard: boolean): void {
    if (this.disabled()) return;
    this.open.set(true);
    const current = this.shown().findIndex((o) => o.value === this.valueText());
    this.activeIndex.set(current >= 0 ? current : fromKeyboard ? 0 : -1);
  }

  close(): void {
    this.open.set(false);
    this.activeIndex.set(-1);
  }

  onBlur(): void {
    this.draft.set(null);
    this.close();
  }

  /** The chevron toggles the list without taking focus from the box. */
  onToggleMousedown(event: MouseEvent): void {
    event.preventDefault();
    if (this.disabled()) return;
    this.field().nativeElement.focus();
    if (this.expanded()) {
      this.close();
    } else {
      this.openList(false);
    }
  }

  /** WAI-ARIA editable combobox with list autocomplete: the arrows move the
   *  highlight, Enter picks it, Escape closes the list. DOM focus stays on the
   *  box throughout, and keys the list does not need do what they would in
   *  any text box. */
  onKeydown(event: KeyboardEvent): void {
    switch (event.key) {
      case 'ArrowDown':
        event.preventDefault();
        if (this.expanded()) {
          this.moveActive(1);
        } else {
          this.openList(true);
        }
        break;
      case 'ArrowUp':
        if (!this.expanded()) return;
        event.preventDefault();
        this.moveActive(-1);
        break;
      case 'Enter':
        if (this.expanded() && this.activeIndex() >= 0) {
          event.preventDefault();
          this.pick(this.shown()[this.activeIndex()]);
        }
        break;
      case 'Escape':
        // Only consume Escape when there is a list to close, so on a closed
        // box it still reaches the dialog behind it.
        if (this.expanded()) {
          event.preventDefault();
          event.stopPropagation();
          this.close();
        }
        break;
      case 'Tab':
        this.close();
        break;
    }
  }

  private moveActive(step: number): void {
    const last = this.shown().length - 1;
    this.activeIndex.set(Math.max(0, Math.min(last, this.activeIndex() + step)));
  }

  private commit(value: string): void {
    if (value !== this.valueText()) this.valueChange.emit(value);
  }

  /**
   * Raise the list into the top layer (once per opening) and hang it under
   * the box at the box's width, flipped above when there is no room below.
   * Fixed to the window like the field-hint tooltip, so no dialog body that
   * clips its overflow can cut it off (#5226).
   */
  private place(list: HTMLElement): void {
    if (typeof list.showPopover === 'function' && !list.matches(':popover-open')) {
      list.showPopover();
    }
    const rect = this.field().nativeElement.getBoundingClientRect();
    list.style.width = `${rect.width}px`;
    const root = document.documentElement;
    const { left, top } = fieldHintPlacement(
      rect,
      { width: rect.width, height: list.offsetHeight },
      { width: root.clientWidth || window.innerWidth, height: root.clientHeight || window.innerHeight },
      'below',
    );
    list.style.left = `${left}px`;
    list.style.top = `${top}px`;
  }
}
