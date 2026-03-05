import { ComponentFixture, TestBed } from '@angular/core/testing';
import { DialogHostComponent } from './dialog-host.component';
import { VtDialogService } from '../../services/dialog.service';

describe('DialogHostComponent', () => {
  let component: DialogHostComponent;
  let fixture: ComponentFixture<DialogHostComponent>;
  let dialogService: VtDialogService;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [DialogHostComponent],
    }).compileComponents();

    fixture = TestBed.createComponent(DialogHostComponent);
    component = fixture.componentInstance;
    dialogService = TestBed.inject(VtDialogService);
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should resolve dialog on button click', async () => {
    const promise = dialogService.confirm('Are you sure?');
    fixture.detectChanges();

    expect(dialogService.dialogOpen).toBeTrue();

    component.onButtonClick(true);
    const result = await promise;
    expect(result).toBeTrue();
    expect(dialogService.dialogOpen).toBeFalse();
  });

  it('should resolve with false on close', async () => {
    const promise = dialogService.confirm('Delete?');
    fixture.detectChanges();

    component.onClosed();
    const result = await promise;
    expect(result).toBeFalse();
  });

  it('should render alert dialog', async () => {
    const promise = dialogService.alert('Something happened');
    fixture.detectChanges();

    expect(dialogService.dialogOpen).toBeTrue();
    expect(dialogService.dialogMessage).toBe('Something happened');

    component.onButtonClick(true);
    await promise;
  });
});
