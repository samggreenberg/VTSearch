import { ComponentFixture, TestBed } from '@angular/core/testing';
import { BrowseLegendComponent } from './browse-legend.component';
import { configureZoneless } from '../../testing/zoneless-testbed';
import { settleZoneless } from '../../testing/settle-resource';

describe('BrowseLegendComponent', () => {
  let fixture: ComponentFixture<BrowseLegendComponent>;
  let priorTheme: string | null;

  beforeEach(async () => {
    priorTheme = document.documentElement.getAttribute('data-theme');
    document.documentElement.removeAttribute('data-theme'); // defaults to dark

    await configureZoneless({
      imports: [BrowseLegendComponent],
    }).compileComponents();

    fixture = TestBed.createComponent(BrowseLegendComponent);
    fixture.componentRef.setInput('maxCount', 100);
    await settleZoneless(fixture);
  });

  afterEach(() => {
    if (priorTheme === null) document.documentElement.removeAttribute('data-theme');
    else document.documentElement.setAttribute('data-theme', priorTheme);
  });

  it('should create', () => {
    expect(fixture.componentInstance).toBeTruthy();
  });

  it('renders the single dot and ramp ticks', () => {
    expect(fixture.nativeElement.querySelector('.browse-legend-dot')).toBeTruthy();
    expect(fixture.nativeElement.querySelectorAll('.browse-legend-ticks .browse-legend-tick').length)
      .toBeGreaterThan(0);
  });

  // Zoneless staleness canary: the live theme is tracked by a `MutationObserver`
  // on `<html data-theme>`. That observer is an unpatched, non-event callback; it
  // only repaints the key because `theme` is a signal read (via the gradient/dot
  // getters) in the template. Flip the document theme and assert the dot's colour
  // changes (auto colormap: HEAT red in dark → OCEAN grey in light) with no
  // manual `detectChanges`.
  it('repaints when the document theme changes (zoneless canary)', async () => {
    const dot = fixture.nativeElement.querySelector('.browse-legend-dot') as HTMLElement;
    const darkColor = dot.style.background;
    expect(darkColor).toBeTruthy();

    document.documentElement.setAttribute('data-theme', 'light');
    await settleZoneless(fixture);

    expect(dot.style.background).not.toBe(darkColor);
  });
});
