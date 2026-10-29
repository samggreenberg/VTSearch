import { ComponentFixture, TestBed } from '@angular/core/testing';
import { DroppedItemsModalComponent } from './dropped-items-modal.component';
import { provideZoneless } from '../../../testing/zoneless-testbed';
import { settleZoneless } from '../../../testing/settle-resource';

describe('DroppedItemsModalComponent', () => {
  let component: DroppedItemsModalComponent;
  let fixture: ComponentFixture<DroppedItemsModalComponent>;

  const groups = [
    { reason: 'The SigLIP embedder returned no vector', count: 5, items: ['a.jpg', 'b.jpg'] },
    { reason: 'Missing the DINOv3 vector the rest of the dataset has', count: 1, items: ['c.jpg'] },
  ];

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [DroppedItemsModalComponent],
      providers: [...provideZoneless()],
    }).compileComponents();

    fixture = TestBed.createComponent(DroppedItemsModalComponent);
    component = fixture.componentInstance;
    fixture.componentRef.setInput('datasetName', 'Photos');
    fixture.componentRef.setInput('groups', groups);
    await settleZoneless(fixture);
  });

  it('lists every recorded item beside its reason', () => {
    const rows = Array.from(fixture.nativeElement.querySelectorAll('.dropped-table tbody tr:not(.truncated-row)')) as HTMLElement[];
    const cells = rows.map((r) => Array.from(r.querySelectorAll('td')).map((td) => td.textContent?.trim()));
    expect(cells).toEqual([
      ['a.jpg', groups[0].reason],
      ['b.jpg', groups[0].reason],
      ['c.jpg', groups[1].reason],
    ]);
  });

  it('counts the items past the name cap rather than hiding them', () => {
    expect(component.total()).toBe(6);
    const truncated = fixture.nativeElement.querySelector('.truncated-row') as HTMLElement;
    expect(truncated.textContent).toContain('3 more not listed');
  });

  it('copies item and reason for every row', () => {
    expect(component.clipboardColumns.map((c) => c.key)).toEqual(['item', 'reason']);
    expect(component.clipboardRows).toEqual([
      { item: 'a.jpg', reason: groups[0].reason },
      { item: 'b.jpg', reason: groups[0].reason },
      { item: 'c.jpg', reason: groups[1].reason },
    ]);
  });

  it('goes back to the Stats window from its Back button', () => {
    const spy = vi.fn();
    component.closed.subscribe(spy);
    (fixture.nativeElement.querySelector('.back-btn') as HTMLButtonElement).click();
    expect(spy).toHaveBeenCalledTimes(1);
  });

  it('says so when nothing was dropped', async () => {
    fixture.componentRef.setInput('groups', []);
    await settleZoneless(fixture);
    expect(fixture.nativeElement.querySelector('.empty-state')).toBeTruthy();
    expect(fixture.nativeElement.querySelector('.dropped-table')).toBeFalsy();
  });
});
