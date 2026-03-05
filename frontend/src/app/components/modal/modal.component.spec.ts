import { ComponentFixture, TestBed } from '@angular/core/testing';
import { Component } from '@angular/core';
import { ModalComponent } from './modal.component';

@Component({
  standalone: true,
  imports: [ModalComponent],
  template: `
    <vt-modal [title]="'Test Modal'" [open]="isOpen" (closed)="onClose()">
      <p>Body content</p>
      <button modal-footer>Footer button</button>
    </vt-modal>
  `,
})
class TestHostComponent {
  isOpen = false;
  closeCalled = false;
  onClose(): void {
    this.closeCalled = true;
  }
}

describe('ModalComponent', () => {
  let fixture: ComponentFixture<TestHostComponent>;
  let host: TestHostComponent;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [TestHostComponent],
    }).compileComponents();
    fixture = TestBed.createComponent(TestHostComponent);
    host = fixture.componentInstance;
  });

  it('should not render when closed', () => {
    host.isOpen = false;
    fixture.detectChanges();
    expect(fixture.nativeElement.querySelector('.modal-backdrop')).toBeNull();
  });

  it('should render when open', () => {
    host.isOpen = true;
    fixture.detectChanges();
    expect(fixture.nativeElement.querySelector('.modal-backdrop')).toBeTruthy();
    expect(fixture.nativeElement.querySelector('.modal-header h2')?.textContent).toContain('Test Modal');
  });

  it('should project body content', () => {
    host.isOpen = true;
    fixture.detectChanges();
    expect(fixture.nativeElement.querySelector('.modal-body p')?.textContent).toContain('Body content');
  });

  it('should emit closed on close button click', () => {
    host.isOpen = true;
    fixture.detectChanges();
    fixture.nativeElement.querySelector('.modal-close').click();
    expect(host.closeCalled).toBeTrue();
  });

  it('should emit closed on backdrop click', () => {
    host.isOpen = true;
    fixture.detectChanges();
    fixture.nativeElement.querySelector('.modal-backdrop').click();
    expect(host.closeCalled).toBeTrue();
  });

  it('should not close on content click', () => {
    host.isOpen = true;
    fixture.detectChanges();
    fixture.nativeElement.querySelector('.modal-content').click();
    expect(host.closeCalled).toBeFalse();
  });
});
