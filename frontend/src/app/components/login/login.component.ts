import { ChangeDetectionStrategy, Component, inject, output, signal } from '@angular/core';

import { AuthService } from '../../services/auth.service';
import { apiErrorMessage } from '../../utils/api-error';

// Deliberately does NOT import FormsModule.  This component and
// `vt-dialog-host` are the only two on the EAGER path, so importing it here
// pulls all of @angular/forms (~37kB) into the initial bundle for one text
// field.  The plain `[value]` + `(input)` pair below is equivalent for a
// single uncontrolled input; use it rather than `ngModel` if you add a field.
// See the bundle-budget note in angular.json and #3811.
@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-login',
  standalone: true,
  templateUrl: './login.component.html',
  styleUrl: './login.component.scss',
})
export class LoginComponent {
  private auth = inject(AuthService);

  readonly loggedIn = output<void>();

  username = '';
  // Signals, not plain fields: the app is zoneless and this component is
  // OnPush, so the async login callbacks are the only writers and nothing else
  // would schedule a repaint. On a failed login that left the form bricked —
  // the input and the submit button stayed `[disabled]="busy"` with no
  // remaining listener to mark the view dirty.
  readonly error = signal('');
  readonly busy = signal(false);

  submit(): void {
    const name = this.username.trim();
    if (!name) {
      this.error.set('Please enter a username.');
      return;
    }
    this.busy.set(true);
    this.error.set('');
    this.auth.login(name).subscribe({
      next: () => {
        this.busy.set(false);
        this.loggedIn.emit();
      },
      error: (err) => {
        this.busy.set(false);
        this.error.set(apiErrorMessage(err, 'Login failed.'));
      },
    });
  }
}
