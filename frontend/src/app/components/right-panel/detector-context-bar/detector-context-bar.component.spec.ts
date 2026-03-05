import { ComponentFixture, TestBed, fakeAsync, tick } from '@angular/core/testing';
import { DetectorContextBarComponent } from './detector-context-bar.component';

describe('DetectorContextBarComponent', () => {
  let component: DetectorContextBarComponent;
  let fixture: ComponentFixture<DetectorContextBarComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [DetectorContextBarComponent],
    }).compileComponents();

    fixture = TestBed.createComponent(DetectorContextBarComponent);
    component = fixture.componentInstance;
  });

  it('should create', () => {
    fixture.detectChanges();
    expect(component).toBeTruthy();
  });

  it('should not render when visible is false', () => {
    component.visible = false;
    fixture.detectChanges();
    const el = fixture.nativeElement as HTMLElement;
    expect(el.querySelector('.train-context-bar')).toBeFalsy();
  });

  it('should render when visible is true', () => {
    component.visible = true;
    component.detectorName = 'My Detector';
    fixture.detectChanges();
    const el = fixture.nativeElement as HTMLElement;
    expect(el.querySelector('.train-context-bar')).toBeTruthy();
    expect(el.querySelector('.train-context-name')?.textContent).toContain('My Detector');
  });

  it('should enter editing mode on startRename', fakeAsync(() => {
    component.visible = true;
    component.detectorName = 'Old Name';
    fixture.detectChanges();

    component.startRename();
    tick();
    fixture.detectChanges();

    expect(component.editing).toBeTrue();
    expect(component.editValue).toBe('Old Name');
  }));

  it('should not start rename if detectorName is empty', () => {
    component.visible = true;
    component.detectorName = '';
    fixture.detectChanges();

    component.startRename();
    expect(component.editing).toBeFalse();
  });

  it('should emit renamed on finishRename with new name', () => {
    spyOn(component.renamed, 'emit');
    component.detectorName = 'Old Name';
    component.editing = true;
    component.editValue = 'New Name';

    component.finishRename();

    expect(component.editing).toBeFalse();
    expect(component.renamed.emit).toHaveBeenCalledWith('New Name');
  });

  it('should not emit renamed if name unchanged', () => {
    spyOn(component.renamed, 'emit');
    component.detectorName = 'Same';
    component.editing = true;
    component.editValue = 'Same';

    component.finishRename();

    expect(component.editing).toBeFalse();
    expect(component.renamed.emit).not.toHaveBeenCalled();
  });

  it('should not emit renamed if name is empty', () => {
    spyOn(component.renamed, 'emit');
    component.detectorName = 'Old';
    component.editing = true;
    component.editValue = '   ';

    component.finishRename();

    expect(component.renamed.emit).not.toHaveBeenCalled();
  });

  it('should cancel rename on Escape', () => {
    component.editing = true;
    component.editValue = 'Something';

    component.onKeydown(new KeyboardEvent('keydown', { key: 'Escape' }));

    expect(component.editing).toBeFalse();
  });

  it('should finish rename on Enter', () => {
    spyOn(component.renamed, 'emit');
    component.detectorName = 'Old';
    component.editing = true;
    component.editValue = 'New';

    component.onKeydown(new KeyboardEvent('keydown', { key: 'Enter' }));

    expect(component.editing).toBeFalse();
    expect(component.renamed.emit).toHaveBeenCalledWith('New');
  });
});
