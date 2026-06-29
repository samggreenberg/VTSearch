import { isTypingTarget, shortcutsBlocked } from './keyboard-shortcuts';

describe('keyboard-shortcuts guards', () => {
  afterEach(() => {
    document.body.innerHTML = '';
  });

  describe('isTypingTarget', () => {
    it('is false for null', () => {
      expect(isTypingTarget(null)).toBe(false);
    });

    it('is true for a text input but false for checkbox/radio/range', () => {
      const text = document.createElement('input');
      text.type = 'text';
      expect(isTypingTarget(text)).toBe(true);

      for (const type of ['checkbox', 'radio', 'range']) {
        const input = document.createElement('input');
        input.type = type;
        expect(isTypingTarget(input)).toBe(false);
      }
    });

    it('is true for textarea, select, and contentEditable', () => {
      expect(isTypingTarget(document.createElement('textarea'))).toBe(true);
      expect(isTypingTarget(document.createElement('select'))).toBe(true);
      const editable = document.createElement('div');
      editable.setAttribute('contenteditable', 'true');
      expect(isTypingTarget(editable)).toBe(true);
    });

    it('is false for a plain element', () => {
      expect(isTypingTarget(document.createElement('div'))).toBe(false);
    });
  });

  describe('shortcutsBlocked', () => {
    it('blocks while a modal backdrop is present', () => {
      const backdrop = document.createElement('div');
      backdrop.className = 'modal-backdrop';
      document.body.appendChild(backdrop);
      expect(shortcutsBlocked()).toBe(true);
    });

    it('blocks while a text field is focused', () => {
      const input = document.createElement('input');
      input.type = 'text';
      document.body.appendChild(input);
      input.focus();
      expect(shortcutsBlocked()).toBe(true);
    });

    it('does not block on a bare page', () => {
      expect(shortcutsBlocked()).toBe(false);
    });
  });
});
