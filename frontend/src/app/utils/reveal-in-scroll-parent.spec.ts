import { revealInScrollParent } from './reveal-in-scroll-parent';

function rect(top: number, height: number): DOMRect {
  return { top, bottom: top + height, height, left: 0, right: 500, width: 500, x: 0, y: top } as DOMRect;
}

/** jsdom lays nothing out, so each box gets a fixed viewport rect and the
 *  target's rect follows the scroller's live `scrollTop`, the way a laid-out
 *  page would. */
function layout(opts: { targetTop: number; targetHeight: number; viewTop?: number; viewHeight?: number }) {
  const outer = document.createElement('div');
  const scroller = document.createElement('div');
  const target = document.createElement('div');
  scroller.style.overflowY = 'auto';
  outer.style.overflowY = 'hidden';
  outer.appendChild(scroller);
  scroller.appendChild(target);
  document.body.appendChild(outer);

  const viewTop = opts.viewTop ?? 100;
  const viewHeight = opts.viewHeight ?? 400;
  let scrollTop = 0;
  Object.defineProperty(scroller, 'scrollTop', {
    get: () => scrollTop,
    set: (v: number) => (scrollTop = v),
  });
  Object.defineProperty(scroller, 'clientHeight', { value: viewHeight });
  Object.defineProperty(scroller, 'scrollHeight', { value: viewHeight * 4 });
  scroller.getBoundingClientRect = () => rect(viewTop, viewHeight);
  target.getBoundingClientRect = () => rect(opts.targetTop - scrollTop, opts.targetHeight);

  return { outer, scroller, target, scrollTop: () => scrollTop };
}

describe('revealInScrollParent', () => {
  afterEach(() => {
    document.body.innerHTML = '';
  });

  it('leaves a target that is already in view alone', () => {
    const { target, scrollTop } = layout({ targetTop: 200, targetHeight: 100 });
    revealInScrollParent(target);
    expect(scrollTop()).toBe(0);
  });

  it('scrolls down just far enough to show a short target below the view', () => {
    // View spans 100..500; target spans 600..700 → 200 px down.
    const { target, scrollTop } = layout({ targetTop: 600, targetHeight: 100 });
    revealInScrollParent(target);
    expect(scrollTop()).toBe(200);
  });

  it('brings the top of a target taller than the view to the top, not its bottom', () => {
    // Target spans 600..1600 in a 400 px view → its top meets the view's top.
    const { target, scrollTop } = layout({ targetTop: 600, targetHeight: 1000 });
    revealInScrollParent(target);
    expect(scrollTop()).toBe(500);
  });

  it('scrolls up to a target above the view', () => {
    const { scroller, target, scrollTop } = layout({ targetTop: 300, targetHeight: 50 });
    scroller.scrollTop = 250; // target now at 50, above the view's top at 100
    revealInScrollParent(target);
    expect(scrollTop()).toBe(200);
  });

  it('never moves an overflow: hidden ancestor', () => {
    const { outer, target } = layout({ targetTop: 600, targetHeight: 1000 });
    revealInScrollParent(target);
    expect(outer.scrollTop).toBe(0);
  });

  it('does nothing when no ancestor scrolls', () => {
    const lone = document.createElement('div');
    document.body.appendChild(lone);
    expect(() => revealInScrollParent(lone)).not.toThrow();
  });
});
