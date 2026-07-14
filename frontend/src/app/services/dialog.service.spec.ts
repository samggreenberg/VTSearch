import { TestBed } from '@angular/core/testing';

import { VtDialogService } from './dialog.service';
import { provideHttpTesting } from '../testing/test-providers';

describe('VtDialogService', () => {
  let service: VtDialogService;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [...provideHttpTesting()],
    });
    service = TestBed.inject(VtDialogService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });

  it('confirm should have Cancel and OK buttons', async () => {
    const promise = service.confirm('Are you sure?');
    expect(service.dialogButtons().length).toBe(2);
    expect(service.dialogButtons()[0].label).toBe('Cancel');
    expect(service.dialogButtons()[1].label).toBe('OK');

    service.resolve(false);
    const result = await promise;
    expect(result).toBe(false);
  });

  it('prompt should return input value', async () => {
    const promise = service.prompt('Enter name', 'default');
    expect(service.dialogShowInput()).toBe(true);
    expect(service.dialogInputValue()).toBe('default');

    service.dialogInputValue.set('typed value');
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
    service.dialogType.set('warning');
    expect(service.getIconType()).toBe('warning');

    service.dialogType.set('error');
    expect(service.getIconType()).toBe('x-circle');

    service.dialogType.set('success');
    expect(service.getIconType()).toBe('check');
  });
});
