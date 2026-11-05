import { Component } from '@angular/core';
import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ColumnChoice, ColumnChooserComponent } from './column-chooser.component';
import { configureZoneless } from '../../testing/zoneless-testbed';
import { settleZoneless } from '../../testing/settle-resource';

describe('ColumnChooserComponent', () => {
  let fixture: ComponentFixture<ColumnChooserComponent>;
  const columns = (...on: boolean[]): ColumnChoice[] =>
    on.map((enabled, i) => ({ key: `c${i}`, label: `Col ${i}`, enabled }));
  const selectAll = () => fixture.nativeElement.querySelector('.column-select-all') as HTMLButtonElement;

  beforeEach(async () => {
    await configureZoneless({ imports: [ColumnChooserComponent] }).compileComponents();
    fixture = TestBed.createComponent(ColumnChooserComponent);
    fixture.componentRef.setInput('columns', columns(true, false));
    await settleZoneless(fixture);
  });

  it('shows a checkbox per column, ticked as the column is', () => {
    const boxes = Array.from(fixture.nativeElement.querySelectorAll('.column-toggle input')) as HTMLInputElement[];
    expect(boxes.map((b) => b.checked)).toEqual([true, false]);
    const labels = Array.from(fixture.nativeElement.querySelectorAll('.column-toggle')) as HTMLElement[];
    expect(labels.map((l) => l.textContent?.trim())).toEqual(['Col 0', 'Col 1']);
  });

  it('reports a box ticked or cleared, leaving the columns to the host', () => {
    const toggled = vi.fn();
    fixture.componentInstance.toggled.subscribe(toggled);
    (fixture.nativeElement.querySelectorAll('.column-toggle input')[1] as HTMLInputElement).click();
    expect(toggled).toHaveBeenCalledWith({ key: 'c1', enabled: true });
  });

  it('marks the select-all as none, some or all', async () => {
    expect(selectAll().getAttribute('aria-checked')).toBe('mixed');
    fixture.componentRef.setInput('columns', columns(true, true));
    await settleZoneless(fixture);
    expect(selectAll().getAttribute('aria-checked')).toBe('true');
    fixture.componentRef.setInput('columns', columns(false, false));
    await settleZoneless(fixture);
    expect(selectAll().getAttribute('aria-checked')).toBe('false');
  });

  it('reports a select-all click', () => {
    const allToggled = vi.fn();
    fixture.componentInstance.allToggled.subscribe(allToggled);
    selectAll().click();
    expect(allToggled).toHaveBeenCalledTimes(1);
  });

  // The host's end-of-row controls (Find Results' thumbnail size buttons,
  // #5584) sit on the checkboxes' row, after the last box.
  it('seats what the host marks chooserEnd at the end of the checkboxes row', async () => {
    @Component({
      standalone: true,
      imports: [ColumnChooserComponent],
      template: `
        <vt-column-chooser [columns]="columns">
          <span class="heading">Columns:</span>
          <button chooserEnd class="sizes" type="button">Bigger</button>
        </vt-column-chooser>
      `,
    })
    class Host {
      columns: ColumnChoice[] = [{ key: 'a', label: 'A', enabled: true }];
    }
    const host = TestBed.createComponent(Host);
    await settleZoneless(host);
    const el: HTMLElement = host.nativeElement;
    expect(el.querySelector('.column-chooser-header .heading')).toBeTruthy();
    const end = el.querySelector('.column-checkboxes .column-chooser-end .sizes');
    expect(end).toBeTruthy();
    expect(el.querySelector('.column-chooser-header .sizes')).toBeNull();
  });
});
