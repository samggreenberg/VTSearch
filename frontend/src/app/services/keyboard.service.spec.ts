import { TestBed } from '@angular/core/testing';
import { KeyboardService, KeyboardAction } from './keyboard.service';

describe('KeyboardService', () => {
  let service: KeyboardService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(KeyboardService);
  });

  afterEach(() => {
    service.stop();
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });

  it('should emit vote good on ArrowRight', (done) => {
    service.start();
    service.action$.subscribe((action: KeyboardAction) => {
      expect(action.type).toBe('vote');
      expect(action.direction).toBe('good');
      done();
    });
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight' }));
  });

  it('should emit vote bad on ArrowLeft', (done) => {
    service.start();
    service.action$.subscribe((action: KeyboardAction) => {
      expect(action.type).toBe('vote');
      expect(action.direction).toBe('bad');
      done();
    });
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowLeft' }));
  });

  it('should emit volume up on ArrowUp', (done) => {
    service.start();
    service.action$.subscribe((action: KeyboardAction) => {
      expect(action.type).toBe('volume');
      expect(action.volumeDelta).toBe(0.05);
      done();
    });
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowUp' }));
  });

  it('should emit volume down on ArrowDown', (done) => {
    service.start();
    service.action$.subscribe((action: KeyboardAction) => {
      expect(action.type).toBe('volume');
      expect(action.volumeDelta).toBe(-0.05);
      done();
    });
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowDown' }));
  });

  it('should emit playback on Space', (done) => {
    service.start();
    service.action$.subscribe((action: KeyboardAction) => {
      expect(action.type).toBe('playback');
      done();
    });
    document.dispatchEvent(new KeyboardEvent('keydown', { key: ' ' }));
  });

  it('should not emit when modifier keys are held', () => {
    service.start();
    const actions: KeyboardAction[] = [];
    service.action$.subscribe((a) => actions.push(a));
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight', ctrlKey: true }));
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight', metaKey: true }));
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight', altKey: true }));
    expect(actions.length).toBe(0);
  });

  it('should not emit when modal backdrop is present', () => {
    service.start();
    const backdrop = document.createElement('div');
    backdrop.classList.add('modal-backdrop');
    document.body.appendChild(backdrop);
    const actions: KeyboardAction[] = [];
    service.action$.subscribe((a) => actions.push(a));
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight' }));
    expect(actions.length).toBe(0);
    document.body.removeChild(backdrop);
  });

  it('should not emit when typing in text input', () => {
    service.start();
    const input = document.createElement('input');
    input.type = 'text';
    document.body.appendChild(input);
    input.focus();
    const actions: KeyboardAction[] = [];
    service.action$.subscribe((a) => actions.push(a));
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight' }));
    expect(actions.length).toBe(0);
    document.body.removeChild(input);
  });

  it('should emit undo on Cmd-Z and Ctrl-Z', () => {
    service.start();
    const actions: KeyboardAction[] = [];
    service.action$.subscribe((a) => actions.push(a));
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'z', metaKey: true }));
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'z', ctrlKey: true }));
    expect(actions.length).toBe(2);
    expect(actions[0].type).toBe('undo');
    expect(actions[1].type).toBe('undo');
  });

  it('should emit redo on Cmd-Shift-Z', () => {
    service.start();
    const actions: KeyboardAction[] = [];
    service.action$.subscribe((a) => actions.push(a));
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Z', metaKey: true, shiftKey: true }));
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'z', ctrlKey: true, shiftKey: true }));
    expect(actions.length).toBe(2);
    expect(actions[0].type).toBe('redo');
    expect(actions[1].type).toBe('redo');
  });

  it('should not emit undo when typing in an input', () => {
    service.start();
    const input = document.createElement('input');
    input.type = 'text';
    document.body.appendChild(input);
    input.focus();
    const actions: KeyboardAction[] = [];
    service.action$.subscribe((a) => actions.push(a));
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'z', metaKey: true }));
    expect(actions.length).toBe(0);
    document.body.removeChild(input);
  });

  it('should not listen after stop()', () => {
    service.start();
    service.stop();
    const actions: KeyboardAction[] = [];
    service.action$.subscribe((a) => actions.push(a));
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight' }));
    expect(actions.length).toBe(0);
  });
});
