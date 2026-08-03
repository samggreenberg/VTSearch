import { ComponentFixture, TestBed } from '@angular/core/testing';
import { JobProgressComponent } from './job-progress.component';
import { provideZoneless } from '../../testing/zoneless-testbed';
import { settleZoneless } from '../../testing/settle-resource';

describe('JobProgressComponent', () => {
  let component: JobProgressComponent;
  let fixture: ComponentFixture<JobProgressComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [JobProgressComponent],
      providers: [...provideZoneless()],
    }).compileComponents();

    fixture = TestBed.createComponent(JobProgressComponent);
    component = fixture.componentInstance;
    await settleZoneless(fixture);
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('renders the header, detail and eta', async () => {
    fixture.componentRef.setInput('header', 'Loading dataset · Step 2 of 4 · Downloading source');
    fixture.componentRef.setInput('detail', '012/345 FileABC.img');
    fixture.componentRef.setInput('eta', 'About 10 min left');
    await settleZoneless(fixture);
    const text = (fixture.nativeElement as HTMLElement).textContent ?? '';
    expect(text).toContain('Downloading source');
    expect(text).toContain('FileABC.img');
    expect(text).toContain('About 10 min left');
  });

  it('labels the eta chip as a whole-job estimate on hover', async () => {
    const el = fixture.nativeElement as HTMLElement;
    // No eta -> no tooltip (an empty chip should not invite hovering).
    expect(el.querySelector('.jp__eta')?.getAttribute('title')).toBeFalsy();
    fixture.componentRef.setInput('eta', 'About 10 min left');
    await settleZoneless(fixture);
    expect(el.querySelector('.jp__eta')?.getAttribute('title')).toContain('whole job');
  });

  it('shows the info chip only when a description is provided', async () => {
    const el = fixture.nativeElement as HTMLElement;
    expect(el.querySelector('.jp__info')).toBeNull();
    fixture.componentRef.setInput('description', 'Fetching the dataset archive.');
    await settleZoneless(fixture);
    const chip = el.querySelector('.jp__info');
    expect(chip).toBeTruthy();
    expect(chip?.getAttribute('title')).toContain('Fetching the dataset archive');
  });

  it('renders Cancel by default and hides it when cancelTitle is null', async () => {
    const el = fixture.nativeElement as HTMLElement;
    expect(el.querySelector('.jp__cancel')).toBeTruthy();
    fixture.componentRef.setInput('cancelTitle', null);
    await settleZoneless(fixture);
    expect(el.querySelector('.jp__cancel')).toBeNull();
  });

  it('passes the bar pulsing state through to the progress bar', async () => {
    fixture.componentRef.setInput('bar', { value: 0.35, max: 1, indeterminate: false, pulsing: true });
    await settleZoneless(fixture);
    const fill = (fixture.nativeElement as HTMLElement).querySelector('.progress-fill');
    expect(fill?.classList).toContain('progress-fill--pulsing');
  });

  it('passes the bar pulseTo bound through as the shimmer band', async () => {
    fixture.componentRef.setInput('bar', {
      value: 0.5,
      max: 1,
      indeterminate: false,
      pulsing: true,
      pulseTo: 0.8,
    });
    await settleZoneless(fixture);
    const band = (fixture.nativeElement as HTMLElement).querySelector(
      '.progress-band',
    ) as HTMLElement;
    expect(band).not.toBeNull();
    expect(band.style.left).toBe('50%');
    expect(band.style.width).toBe('30%');
  });

  it('emits cancel and stops the click from bubbling', async () => {
    const emit = vi.spyOn(component.cancel, 'emit');
    const btn = (fixture.nativeElement as HTMLElement).querySelector(
      '.jp__cancel',
    ) as HTMLButtonElement;
    const event = new MouseEvent('click', { bubbles: true });
    const stop = vi.spyOn(event, 'stopPropagation');
    btn.dispatchEvent(event);
    expect(emit).toHaveBeenCalled();
    expect(stop).toHaveBeenCalled();
  });

  it('swaps the Cancel button for a disabled "Cancelling…" badge and detail when cancelling', async () => {
    const el = fixture.nativeElement as HTMLElement;
    fixture.componentRef.setInput('detail', '012/345 FileABC.img');
    await settleZoneless(fixture);
    // Before: live, enabled Cancel button.
    let btn = el.querySelector('.jp__cancel') as HTMLButtonElement;
    expect(btn.disabled).toBe(false);
    expect(btn.textContent?.trim()).toBe('Cancel');
    expect(el.querySelector('.jp__detail')?.textContent?.trim()).toBe('012/345 FileABC.img');

    fixture.componentRef.setInput('cancelling', true);
    await settleZoneless(fixture);

    // After: disabled acknowledgement badge, and the detail reads "Cancelling…".
    btn = el.querySelector('.jp__cancel') as HTMLButtonElement;
    expect(btn.disabled).toBe(true);
    expect(btn.textContent?.trim()).toBe('Cancelling…');
    expect(el.querySelector('.jp__detail')?.textContent?.trim()).toBe('Cancelling…');
  });

  it('does not emit cancel while cancelling (badge has no click handler)', async () => {
    const emit = vi.spyOn(component.cancel, 'emit');
    fixture.componentRef.setInput('cancelling', true);
    await settleZoneless(fixture);
    const btn = (fixture.nativeElement as HTMLElement).querySelector(
      '.jp__cancel',
    ) as HTMLButtonElement;
    btn.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    expect(emit).not.toHaveBeenCalled();
  });

  it('adds the cell host class in table-cell mode', async () => {
    expect((fixture.nativeElement as HTMLElement).classList.contains('jp-host--cell')).toBe(false);
    fixture.componentRef.setInput('cell', true);
    await settleZoneless(fixture);
    expect((fixture.nativeElement as HTMLElement).classList.contains('jp-host--cell')).toBe(true);
  });
});
