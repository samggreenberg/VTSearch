import { ComponentFixture, TestBed } from '@angular/core/testing';
import { AutoDetectProgressModalComponent } from './autodetect-progress-modal.component';

describe('AutoDetectProgressModalComponent', () => {
  let component: AutoDetectProgressModalComponent;
  let fixture: ComponentFixture<AutoDetectProgressModalComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [AutoDetectProgressModalComponent],
    }).compileComponents();

    fixture = TestBed.createComponent(AutoDetectProgressModalComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should display progress percentage', () => {
    component.progress = 75;
    component.statusText = 'Running detector 2 of 3...';
    fixture.detectChanges();
    const el = fixture.nativeElement as HTMLElement;
    expect(el.textContent).toContain('75%');
    expect(el.textContent).toContain('Running detector 2 of 3');
  });

  it('should emit cancelled on cancel click', () => {
    spyOn(component.cancelled, 'emit');
    const btn = fixture.nativeElement.querySelector('button');
    btn.click();
    expect(component.cancelled.emit).toHaveBeenCalled();
  });

  it('should emit closed on close', () => {
    spyOn(component.closed, 'emit');
    component.close();
    expect(component.closed.emit).toHaveBeenCalled();
  });
});
