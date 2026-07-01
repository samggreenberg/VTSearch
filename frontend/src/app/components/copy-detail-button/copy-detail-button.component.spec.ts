import { ComponentFixture, TestBed } from '@angular/core/testing';
import { CopyDetailButtonComponent } from './copy-detail-button.component';
import { configureZoneless } from '../../testing/zoneless-testbed';
import { settleZoneless } from '../../testing/settle-resource';

describe('CopyDetailButtonComponent', () => {
  let fixture: ComponentFixture<CopyDetailButtonComponent>;
  let written: string[];

  beforeEach(async () => {
    written = [];
    Object.defineProperty(navigator, 'clipboard', {
      value: {
        writeText: (t: string) => {
          written.push(t);
          return Promise.resolve();
        },
      },
      configurable: true,
    });

    await configureZoneless({
      imports: [CopyDetailButtonComponent],
    }).compileComponents();

    fixture = TestBed.createComponent(CopyDetailButtonComponent);
    fixture.componentRef.setInput('value', 'abc123');
    fixture.componentRef.setInput('label', 'MD5');
    await settleZoneless(fixture);
  });

  it('should create', () => {
    expect(fixture.componentInstance).toBeTruthy();
  });

  it('titles the button with the category name', () => {
    const btn = fixture.nativeElement.querySelector('button') as HTMLButtonElement;
    expect(btn.getAttribute('title')).toBe('Copy MD5 to clipboard');
    expect(btn.getAttribute('aria-label')).toBe('Copy MD5 to clipboard');
  });

  // Zoneless canary: `copied` flips in the post-`await` continuation of the
  // click handler, outside the click's CD-scheduling stack. It only repaints
  // because `copied` is a signal.
  it('writes the value and flashes the copied state after the async write', async () => {
    const btn = fixture.nativeElement.querySelector('button') as HTMLButtonElement;
    btn.click();
    await settleZoneless(fixture);
    expect(written).toEqual(['abc123']);
    expect(fixture.componentInstance.copied()).toBe(true);
    expect(btn.classList).toContain('copied');
  });
});
