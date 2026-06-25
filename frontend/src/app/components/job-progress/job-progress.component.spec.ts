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
    fixture.componentRef.setInput('header', 'Loading dataset · step 2 of 4 · downloading source');
    fixture.componentRef.setInput('detail', '(012/345) FileABC.img');
    fixture.componentRef.setInput('eta', '~5.5 min left');
    await settleZoneless(fixture);
    const text = (fixture.nativeElement as HTMLElement).textContent ?? '';
    expect(text).toContain('downloading source');
    expect(text).toContain('FileABC.img');
    expect(text).toContain('~5.5 min left');
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

  it('adds the cell host class in table-cell mode', async () => {
    expect((fixture.nativeElement as HTMLElement).classList.contains('jp-host--cell')).toBe(false);
    fixture.componentRef.setInput('cell', true);
    await settleZoneless(fixture);
    expect((fixture.nativeElement as HTMLElement).classList.contains('jp-host--cell')).toBe(true);
  });
});
