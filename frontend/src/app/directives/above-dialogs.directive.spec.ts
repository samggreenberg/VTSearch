import { ComponentFixture, TestBed } from '@angular/core/testing';
import { Component, signal } from '@angular/core';

import { AboveDialogsDirective } from './above-dialogs.directive';
import { ModalComponent } from '../components/modal/modal.component';
import { provideZoneless } from '../testing/zoneless-testbed';
import { settleZoneless } from '../testing/settle-resource';

@Component({
  standalone: true,
  imports: [AboveDialogsDirective, ModalComponent],
  template: `
    <div class="before"></div>
    <div class="overlay" vtAboveDialogs></div>
    <div class="after"></div>
    @if (outerOpen()) {
      <vt-modal title="Outer" [open]="true">
        <p>Outer</p>
        @if (innerOpen()) {
          <vt-modal title="Inner" [open]="true"><p>Inner</p></vt-modal>
        }
      </vt-modal>
    }
  `,
})
class HostComponent {
  readonly outerOpen = signal(false);
  readonly innerOpen = signal(false);
}

describe('AboveDialogsDirective', () => {
  let fixture: ComponentFixture<HostComponent>;
  let host: HostComponent;
  let overlay: HTMLElement;

  const dialogs = () => [...fixture.nativeElement.querySelectorAll('dialog')] as HTMLDialogElement[];

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [HostComponent],
      providers: [...provideZoneless()],
    }).compileComponents();
    fixture = TestBed.createComponent(HostComponent);
    host = fixture.componentInstance;
    await settleZoneless(fixture);
    overlay = fixture.nativeElement.querySelector('.overlay');
  });

  function expectAtHome(): void {
    // Exactly where the template put it, with no marker left behind.
    expect(overlay.previousElementSibling?.className).toBe('before');
    expect(overlay.nextElementSibling?.className).toBe('after');
    expect(overlay.previousSibling?.nodeType).not.toBe(Node.COMMENT_NODE);
  }

  it('stays where it is while no dialog is open', () => {
    expectAtHome();
  });

  it('moves into the open dialog, outside of which the page is inert, and back when it closes', async () => {
    host.outerOpen.set(true);
    await settleZoneless(fixture);
    expect(overlay.parentElement).toBe(dialogs()[0]);

    host.outerOpen.set(false);
    await settleZoneless(fixture);
    expectAtHome();
  });

  it('follows the topmost dialog through a stack', async () => {
    host.outerOpen.set(true);
    await settleZoneless(fixture);
    host.innerOpen.set(true);
    await settleZoneless(fixture);
    const [outer, inner] = dialogs();
    expect(overlay.parentElement).toBe(inner);

    // The inner dialog goes, taking the overlay out of the document with it;
    // it lands in the outer one, not nowhere.
    host.innerOpen.set(false);
    await settleZoneless(fixture);
    expect(overlay.parentElement).toBe(outer);
    expect(overlay.isConnected).toBe(true);

    host.outerOpen.set(false);
    await settleZoneless(fixture);
    expectAtHome();
  });
});
