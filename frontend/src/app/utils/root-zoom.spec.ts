import { readRootZoom } from './root-zoom';

describe('readRootZoom', () => {
  let original: typeof window.getComputedStyle;

  function stubZoom(value: string): void {
    (window as unknown as { getComputedStyle: (el: Element) => CSSStyleDeclaration }).getComputedStyle =
      (el: Element) => {
        if (el === document.documentElement) {
          return { zoom: value } as unknown as CSSStyleDeclaration;
        }
        return original(el);
      };
  }

  beforeEach(() => {
    original = window.getComputedStyle;
  });

  afterEach(() => {
    (window as unknown as { getComputedStyle: typeof window.getComputedStyle }).getComputedStyle = original;
  });

  it('reads a fractional zoom factor off the root element', () => {
    stubZoom('1.1');
    expect(readRootZoom()).toBeCloseTo(1.1, 5);
  });

  it('reads an integer zoom factor', () => {
    stubZoom('2');
    expect(readRootZoom()).toBe(2);
  });

  it('falls back to 1 when the computed zoom is the default "normal"', () => {
    stubZoom('normal');
    expect(readRootZoom()).toBe(1);
  });

  it('falls back to 1 when the computed zoom is empty', () => {
    stubZoom('');
    expect(readRootZoom()).toBe(1);
  });

  it('falls back to 1 for a zero computed zoom that would divide-by-zero downstream', () => {
    stubZoom('0');
    expect(readRootZoom()).toBe(1);
  });
});
