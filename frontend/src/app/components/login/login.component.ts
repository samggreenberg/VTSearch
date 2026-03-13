import { Component, EventEmitter, Output } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { AuthService } from '../../services/auth.service';

@Component({
  selector: 'vt-login',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './login.component.html',
  styleUrl: './login.component.scss',
})
export class LoginComponent {
  @Output() loggedIn = new EventEmitter<void>();

  username = '';
  error = '';
  busy = false;

  constructor(private auth: AuthService) {}

  submit(): void {
    const name = this.username.trim();
    if (!name) {
      this.error = 'Please enter a username.';
      return;
    }
    this.busy = true;
    this.error = '';
    this.auth.login(name).subscribe({
      next: () => {
        this.busy = false;
        this.loggedIn.emit();
      },
      error: (err) => {
        this.busy = false;
        this.error = err.error?.error || 'Login failed.';
      },
    });
  }
}
