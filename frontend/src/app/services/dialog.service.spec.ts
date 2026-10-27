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

  // #4380: the question and its detail used to be joined with a space, so a
  // destructive confirmation read as one run-on line.
  it('confirmDestructive puts the detail on its own line', async () => {
    const promise = service.confirmDestructive("Delete detector 'cats'?", '(Media is unaffected.)');
    expect(service.dialogMessage()).toBe("Delete detector 'cats'?\n(Media is unaffected.)");
    service.cancel();
    expect(await promise).toBe(false);
  });

  it('confirmDestructiveWithEscape puts the detail on its own line', async () => {
    const promise = service.confirmDestructiveWithEscape('Reset?', 'Cannot be undone.', 'Reset', 'Export first…');
    expect(service.dialogMessage()).toBe('Reset?\nCannot be undone.');
    service.cancel();
    expect(await promise).toBe('cancel');
  });

  it('confirmDestructiveWithEscape exposes Cancel, escape, and action buttons', async () => {
    const promise = service.confirmDestructiveWithEscape('Reset?', 'Cannot be undone.', 'Reset', 'Export first…');
    const labels = service.dialogButtons().map((b) => b.label);
    expect(labels).toEqual(['Cancel', 'Export first…', 'Reset']);
    // Cancel is the first non-primary button, so backdrop/Escape dismissal resolves 'cancel'.
    service.cancel();
    expect(await promise).toBe('cancel');
  });

  it('confirmDestructiveWithEscape resolves escape when the escape hatch is chosen', async () => {
    const promise = service.confirmDestructiveWithEscape('Reset?', 'Cannot be undone.', 'Reset', 'Export first…');
    service.resolve('escape');
    expect(await promise).toBe('escape');
  });

  it('choose shows its own buttons and resolves the clicked one', async () => {
    const promise = service.choose('Where to?', [
      { label: 'Not now', primary: false, value: 'later' },
      { label: 'Go', primary: true, value: 'go' },
    ]);
    expect(service.dialogMessage()).toBe('Where to?');
    expect(service.dialogType()).toBe('info');
    expect(service.dialogButtons().map((b) => b.label)).toEqual(['Not now', 'Go']);
    service.resolve('go');
    expect(await promise).toBe('go');
  });

  it('choose resolves the first non-primary answer on dismissal', async () => {
    const promise = service.choose('Where to?', [
      { label: 'Not now', primary: false, value: 'later' },
      { label: 'Elsewhere', primary: false, value: 'elsewhere' },
      { label: 'Go', primary: true, value: 'go' },
    ]);
    service.cancel();
    expect(await promise).toBe('later');
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
