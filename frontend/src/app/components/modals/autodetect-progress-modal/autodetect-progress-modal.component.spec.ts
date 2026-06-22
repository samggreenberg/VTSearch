import { ComponentFixture, TestBed } from '@angular/core/testing';
import { AutoDetectProgressModalComponent } from './autodetect-progress-modal.component';
import { provideZoneless } from '../../../testing/zoneless-testbed';
import { settleZoneless } from '../../../testing/settle-resource';

describe('AutoDetectProgressModalComponent', () => {
  let component: AutoDetectProgressModalComponent;
  let fixture: ComponentFixture<AutoDetectProgressModalComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [AutoDetectProgressModalComponent],
      providers: [...provideZoneless()],
    }).compileComponents();

    fixture = TestBed.createComponent(AutoDetectProgressModalComponent);
    component = fixture.componentInstance;
    await settleZoneless(fixture);
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should display progress percentage', async () => {
    fixture.componentRef.setInput('progress', 75);
    fixture.componentRef.setInput('statusText', 'Running detector 2 of 3...');
    await settleZoneless(fixture);
    const el = fixture.nativeElement as HTMLElement;
    expect(el.textContent).toContain('75%');
    expect(el.textContent).toContain('Running detector 2 of 3');
  });

  it('should emit cancelled on cancel click', () => {
    vi.spyOn(component.cancelled, 'emit');
    // The modal chrome renders its own close (X) button first, so target the
    // Cancel button specifically by its title.
    const btn = fixture.nativeElement.querySelector(
      'button[title="Cancel the auto-detect process"]',
    ) as HTMLButtonElement;
    btn.click();
    expect(component.cancelled.emit).toHaveBeenCalled();
  });

  it('should emit closed on close', () => {
    vi.spyOn(component.closed, 'emit');
    component.close();
    expect(component.closed.emit).toHaveBeenCalled();
  });
});
