import { ChangeDetectionStrategy, Component, inject, output } from '@angular/core';

import { FormsModule } from '@angular/forms';
import { AuthService } from '../../services/auth.service';
import { apiErrorMessage } from '../../utils/api-error';

@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-login',
  standalone: true,
  imports: [FormsModule],
  templateUrl: './login.component.html',
  styleUrl: './login.component.scss',
})
export class LoginComponent {
  private auth = inject(AuthService);

  readonly loggedIn = output<void>();

  username = '';
  error = '';
  busy = false;

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
        this.error = apiErrorMessage(err, 'Login failed.');
      },
    });
  }
}
