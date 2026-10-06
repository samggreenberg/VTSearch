import { ComponentFixture, TestBed } from '@angular/core/testing';

import { LineTestPanelComponent } from './line-test-panel.component';
import { provideZoneless } from '../../../testing/zoneless-testbed';
import { settleZoneless } from '../../../testing/settle-resource';
import { wireDone, wireLineTest, wireTest } from '../../../testing/line-test';
import type { LineTestResponse } from '../../../generated/api-client/models/line-test-response';

describe('LineTestPanelComponent (#4524)', () => {
  let fixture: ComponentFixture<LineTestPanelComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [LineTestPanelComponent],
      providers: [...provideZoneless()],
    }).compileComponents();
    fixture = TestBed.createComponent(LineTestPanelComponent);
    await settleZoneless(fixture);
  });

  async function show(response: LineTestResponse | null, scoring = false, scoreProgress: number | null = null): Promise<HTMLElement> {
    fixture.componentRef.setInput('response', response);
    fixture.componentRef.setInput('scoring', scoring);
    fixture.componentRef.setInput('scoreProgress', scoreProgress);
    await settleZoneless(fixture);
    return fixture.nativeElement as HTMLElement;
  }

  const steps = (el: HTMLElement) => Array.from(el.querySelectorAll('.ap-step')) as HTMLElement[];
  const active = (el: HTMLElement) => el.querySelector('.ap-step.active') as HTMLElement;
  const light = (el: HTMLElement) => active(el).querySelector('.ap-light') as HTMLElement | null;

  it('lists the four steps: Score, Check the matches, Check the misses, Done', async () => {
    const el = await show(null, true);
    expect(steps(el).map((s) => s.querySelector('.ap-step-label')!.textContent!.trim())).toEqual([
      'Score.',
      'Check the matches.',
      'Check the misses.',
      'Done!',
    ]);
  });

  it('Score is active while the pass runs, its light the pass\'s progress', async () => {
    let el = await show(null, true, 0.2);
    expect(active(el).dataset['phase']).toBe('score');
    expect(light(el)!.dataset['color']).toBe('red');
    expect(active(el).querySelector('.ap-step-detail')!.textContent).toContain('20% scored');
    el = await show(null, true, 0.7);
    expect(light(el)!.dataset['color']).toBe('yellow');
  });

  it('Check the matches is active with a running test, its light the precision range\'s width against its target', async () => {
    const wide = wireTest();
    let el = await show(wireLineTest(wide));
    expect(active(el).dataset['phase']).toBe('matches');
    expect(steps(el)[0].classList.contains('done')).toBe(true);
    // 0.6 wide against a 0.2 target: beyond twice it.
    expect(light(el)!.dataset['color']).toBe('red');
    expect(active(el).querySelector('.ap-step-detail')!.textContent).toContain('likely 55–95% right');

    const narrower = wireTest();
    narrower.report = { ...narrower.report, matches_width: 0.3, picks_above: 10 };
    el = await show(wireLineTest(narrower));
    expect(light(el)!.dataset['color']).toBe('yellow');
    expect(active(el).querySelector('.ap-step-detail')!.textContent).toContain('10 picks');
  });

  it('Check the misses tracks the recall range\'s width and reads the found share in words', async () => {
    const t = wireTest({ phase: 'misses' });
    t.report = { ...t.report, phase: 'misses', matches_stop: 'width', misses_width: 0.2, picks_below: 5 };
    const el = await show(wireLineTest(t));
    expect(active(el).dataset['phase']).toBe('misses');
    expect(light(el)).toBeNull();
    expect(active(el).querySelector('.ap-check')).not.toBeNull();
    expect(active(el).querySelector('.ap-step-detail')!.textContent).toContain('about half of them found');
  });

  it('says how Check the misses ends: at its budget with a class model, at a dry band without one (#4542)', async () => {
    const t = wireTest({ phase: 'misses' });
    t.report = { ...t.report, phase: 'misses', matches_stop: 'width', misses_width: 0.2, picks_below: 5 };
    let el = await show(wireLineTest(t));
    let title = active(el).querySelector('.ap-check')!.getAttribute('title')!;
    expect(title).toContain('ends at 40 picks, or once every band below the line is checked, however narrow the range');
    expect(title).not.toContain('turns up nothing');
    el = await show(wireLineTest({ ...t, class_model: false }));
    title = active(el).querySelector('.ap-check')!.getAttribute('title')!;
    expect(title).toContain('when the step ends. It also ends at 40 picks, or when a band below the line turns up nothing.');
  });

  it('Done is green with the picks behind it', async () => {
    const el = await show(wireLineTest(wireDone()));
    expect(active(el).dataset['phase']).toBe('done');
    expect(active(el).querySelector('.ap-check')).not.toBeNull();
    expect(active(el).querySelector('.ap-step-detail')!.textContent).toContain('65 picks');
    expect(steps(el).filter((s) => s.classList.contains('done')).length).toBe(3);
  });

  it('says there is nothing to test in place of the steps', async () => {
    const el = await show(wireLineTest(wireTest({ phase: 'nothing', picks: [] })));
    expect(el.querySelector('.ap-step')).toBeNull();
    expect(el.querySelector('.ap-exhausted-title')!.textContent).toContain('Nothing to test');
  });
});
