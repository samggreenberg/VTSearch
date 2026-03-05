import { Component } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { DialogHostComponent } from './components/dialog-host/dialog-host.component';

@Component({
  selector: 'app-root',
  imports: [RouterOutlet, DialogHostComponent],
  templateUrl: './app.component.html',
  styleUrl: './app.component.scss',
})
export class AppComponent {
  title = 'VTSearch';
}
