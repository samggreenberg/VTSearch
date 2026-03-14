import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { provideHttpClientTesting } from '@angular/common/http/testing';
import { VotingOverlayComponent } from './voting-overlay.component';

describe('VotingOverlayComponent', () => {
  let component: VotingOverlayComponent;
  let fixture: ComponentFixture<VotingOverlayComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [VotingOverlayComponent],
      providers: [provideHttpClient(), provideHttpClientTesting()],
    }).compileComponents();
    fixture = TestBed.createComponent(VotingOverlayComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should render Good and Bad vote buttons', () => {
    const voteButtons = fixture.nativeElement.querySelectorAll('.vote-buttons button');
    expect(voteButtons.length).toBe(2);
    expect(voteButtons[0].textContent.trim()).toBe('Bad');
    expect(voteButtons[1].textContent.trim()).toBe('Good');
  });

  it('should render Add media to Good and Add media to Bad buttons', () => {
    const addButtons = fixture.nativeElement.querySelectorAll('.add-media-buttons button');
    expect(addButtons.length).toBe(2);
    expect(addButtons[0].textContent.trim()).toBe('Add media to Bad');
    expect(addButtons[1].textContent.trim()).toBe('Add media to Good');
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

  it('should have hidden file inputs for add-to-pile', () => {
    const hiddenInputs = fixture.nativeElement.querySelectorAll('.hidden-file-input');
    expect(hiddenInputs.length).toBe(2);
  });
});
