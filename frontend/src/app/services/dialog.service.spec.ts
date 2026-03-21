import { TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { provideHttpClientTesting } from '@angular/common/http/testing';
import { VtDialogService } from './dialog.service';

describe('VtDialogService', () => {
  let service: VtDialogService;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [provideHttpClient(), provideHttpClientTesting()],
    });
    service = TestBed.inject(VtDialogService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });

  it('alert should open dialog and resolve on OK', async () => {
    const promise = service.alert('Hello');
    expect(service.dialogOpen).toBeTrue();
    expect(service.dialogMessage).toBe('Hello');
    expect(service.dialogType).toBe('info');
    expect(service.dialogButtons.length).toBe(1);

    service.resolve(true);
    const result = await promise;
    expect(result).toBeTrue();
    expect(service.dialogOpen).toBeFalse();
  });

  it('confirm should have Cancel and OK buttons', async () => {
    const promise = service.confirm('Are you sure?');
    expect(service.dialogButtons.length).toBe(2);
    expect(service.dialogButtons[0].label).toBe('Cancel');
    expect(service.dialogButtons[1].label).toBe('OK');

    service.resolve(false);
    const result = await promise;
    expect(result).toBeFalse();
  });

  it('prompt should return input value', async () => {
    const promise = service.prompt('Enter name', 'default');
    expect(service.dialogShowInput).toBeTrue();
    expect(service.dialogInputValue).toBe('default');

    service.dialogInputValue = 'typed value';
    service.resolve('__input__');
    const result = await promise;
    expect(result).toBe('typed value');
  });

  it('prompt cancel should return null', async () => {
    const promise = service.prompt('Enter');
    service.resolve(null);
    const result = await promise;
    expect(result).toBeNull();
  });

  it('getIconType should return correct icon type', () => {
    service.dialogType = 'warning';
    expect(service.getIconType()).toBe('warning');

    service.dialogType = 'error';
    expect(service.getIconType()).toBe('x-circle');

    service.dialogType = 'success';
    expect(service.getIconType()).toBe('check');
  });
});
