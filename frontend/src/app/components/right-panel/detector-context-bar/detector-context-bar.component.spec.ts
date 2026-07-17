import { ComponentFixture, TestBed } from '@angular/core/testing';
import { DetectorContextBarComponent } from './detector-context-bar.component';
import { provideZoneless } from '../../../testing/zoneless-testbed';
import { settleZoneless } from '../../../testing/settle-resource';

describe('DetectorContextBarComponent', () => {
  let component: DetectorContextBarComponent;
  let fixture: ComponentFixture<DetectorContextBarComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [DetectorContextBarComponent],
      providers: [...provideZoneless()],
    }).compileComponents();

    fixture = TestBed.createComponent(DetectorContextBarComponent);
    component = fixture.componentInstance;
  });

  it('should create', async () => {
    await settleZoneless(fixture);
    expect(component).toBeTruthy();
  });

  it('should not render when visible is false', async () => {
    fixture.componentRef.setInput('visible', false);
    await settleZoneless(fixture);
    const el = fixture.nativeElement as HTMLElement;
    expect(el.querySelector('.train-context-bar')).toBeFalsy();
  });

  it('should render when visible is true', async () => {
    fixture.componentRef.setInput('visible', true);
    fixture.componentRef.setInput('detectorName', 'My Detector');
    await settleZoneless(fixture);
    const el = fixture.nativeElement as HTMLElement;
    expect(el.querySelector('.train-context-bar')).toBeTruthy();
    expect(el.querySelector('.train-context-name')?.textContent).toContain('My Detector');
  });

  it('should enter editing mode on startRename', async () => {
    fixture.componentRef.setInput('visible', true);
    fixture.componentRef.setInput('detectorName', 'Old Name');
    await settleZoneless(fixture);

    component.startRename();
    // Drain the focus setTimeout queued by startRename(), then settle so the
    // editing input renders.
    await settleZoneless(fixture);

    expect(component.editing).toBe(true);
    expect(component.editValue).toBe('Old Name');
  });

  it('should not start rename if detectorName is empty', async () => {
    fixture.componentRef.setInput('visible', true);
    fixture.componentRef.setInput('detectorName', '');
    await settleZoneless(fixture);

    component.startRename();
    expect(component.editing).toBe(false);
  });

  it('should emit renamed on finishRename with new name', () => {
    vi.spyOn(component.renamed, 'emit');
    fixture.componentRef.setInput('detectorName', 'Old Name');
    component.editing = true;
    component.editValue = 'New Name';

    component.finishRename();

    expect(component.editing).toBe(false);
    expect(component.renamed.emit).toHaveBeenCalledWith('New Name');
  });

  it('should not emit renamed if name unchanged', () => {
    vi.spyOn(component.renamed, 'emit');
    fixture.componentRef.setInput('detectorName', 'Same');
    component.editing = true;
    component.editValue = 'Same';

    component.finishRename();

    expect(component.editing).toBe(false);
    expect(component.renamed.emit).not.toHaveBeenCalled();
  });

  it('should not emit renamed if name is empty', () => {
    vi.spyOn(component.renamed, 'emit');
    fixture.componentRef.setInput('detectorName', 'Old');
    component.editing = true;
    component.editValue = '   ';

    component.finishRename();

    expect(component.renamed.emit).not.toHaveBeenCalled();
  });

  it('should cancel rename on Escape', () => {
    component.editing = true;
    component.editValue = 'Something';

    component.onKeydown(new KeyboardEvent('keydown', { key: 'Escape' }));

    expect(component.editing).toBe(false);
  });

  it('should finish rename on Enter', () => {
    vi.spyOn(component.renamed, 'emit');
    fixture.componentRef.setInput('detectorName', 'Old');
    component.editing = true;
    component.editValue = 'New';

    component.onKeydown(new KeyboardEvent('keydown', { key: 'Enter' }));

    expect(component.editing).toBe(false);
    expect(component.renamed.emit).toHaveBeenCalledWith('New');
  });
});
