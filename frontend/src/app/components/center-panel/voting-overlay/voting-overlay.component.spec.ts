import { ComponentFixture, TestBed } from '@angular/core/testing';
import { VotingOverlayComponent } from './voting-overlay.component';

describe('VotingOverlayComponent', () => {
  let component: VotingOverlayComponent;
  let fixture: ComponentFixture<VotingOverlayComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [VotingOverlayComponent],
    }).compileComponents();
    fixture = TestBed.createComponent(VotingOverlayComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should render Good and Bad buttons', () => {
    const buttons = fixture.nativeElement.querySelectorAll('button');
    expect(buttons.length).toBe(2);
    expect(buttons[0].textContent.trim()).toBe('Bad');
    expect(buttons[1].textContent.trim()).toBe('Good');
  });

  it('should emit good on Good click', () => {
    let emitted: string | undefined;
    component.voted.subscribe((v: string) => (emitted = v));
    fixture.nativeElement.querySelector('.btn-good').click();
    expect(emitted).toBe('good');
  });

  it('should emit bad on Bad click', () => {
    let emitted: string | undefined;
    component.voted.subscribe((v: string) => (emitted = v));
    fixture.nativeElement.querySelector('.btn-bad').click();
    expect(emitted).toBe('bad');
  });

  it('should apply voted class when isGood is true', () => {
    component.isGood = true;
    fixture.detectChanges();
    expect(fixture.nativeElement.querySelector('.btn-good').classList.contains('voted')).toBeTrue();
  });

  it('should apply voted class when isBad is true', () => {
    component.isBad = true;
    fixture.detectChanges();
    expect(fixture.nativeElement.querySelector('.btn-bad').classList.contains('voted')).toBeTrue();
  });

  it('should not emit when disabled', () => {
    component.disabled = true;
    fixture.detectChanges();
    let emitted = false;
    component.voted.subscribe(() => (emitted = true));
    fixture.nativeElement.querySelector('.btn-good').click();
    expect(emitted).toBeFalse();
  });

  it('should hide the first-vote hint by default', () => {
    expect(fixture.nativeElement.querySelector('.vote-hint')).toBeNull();
  });

  it('should render the first-vote hint text when showHint is true', () => {
    component.showHint = true;
    fixture.detectChanges();
    const hint = fixture.nativeElement.querySelector('.vote-hint');
    expect(hint).toBeTruthy();
    expect(hint.textContent.trim()).toContain('Use');
    expect(hint.textContent).toContain('Autopilot');
  });
});
