# VTSearch Frontend

Angular SPA for the VTSearch media explorer. Built with Angular CLI 19.2 and TypeScript 5.7.

## Prerequisites

- **Node.js 18+** (LTS recommended)
- **npm** (bundled with Node.js)

## Development server

Start the Flask backend first (`python app.py --local` from the project root), then:

```bash
npm install   # first time only
npm start
```

This starts the Angular dev server at `http://localhost:4200/` with a proxy that forwards `/api/*` requests to the Flask backend at `localhost:5000` (configured in `proxy.conf.json`). The application automatically reloads when source files change.

## Building for production

```bash
npm run build:prod
```

This compiles the Angular app and outputs the build artifacts to `../static/` (the project root's `static/` directory), where Flask serves them. Output files: `index.html`, `main.js`, `polyfills.js`, `styles.css`.

## Project structure

```
src/app/
├── app.component.ts          # Root component
├── app.routes.ts              # Route definitions
├── components/                # UI components
│   ├── center-panel/          # Media viewer (image, text, video, audio, document)
│   ├── left-panel/            # Media list, sort bar, inclusion slider, autopilot
│   ├── right-panel/           # Labels and detector context
│   ├── dashboard/             # Dataset and model management UI
│   ├── label-view/            # Main labeling view (orchestrates left/center/right panels)
│   ├── find-view/             # Multi-dataset search interface
│   ├── login/                 # Authentication screen
│   ├── modals/                # 17 modal dialogs (export, import, settings, progress, etc.)
│   ├── dialog-host/           # Modal container
│   ├── file-browser/          # Server file picker
│   ├── progress-bar/          # Progress indicators
│   └── icon/                  # Icon system
├── services/                  # State management and API communication
│   ├── *-api.service.ts       # API services (one per backend module)
│   ├── *-state.service.ts     # Client-side state services
│   ├── active-context.service.ts  # Tracks selected dataset/detector
│   ├── dialog.service.ts      # Modal management
│   ├── keyboard.service.ts    # Keyboard shortcuts
│   └── theme.service.ts       # Theme (light/dark) switching
├── interceptors/
│   └── active-context.interceptor.ts  # Attaches X-Dataset-Id/X-Detector-Id headers
└── models/
    └── api.models.ts          # TypeScript interfaces for API responses
```

### Key architecture patterns

- **ActiveContextService** tracks which dataset and detector the user has selected. The `activeContextInterceptor` attaches `X-Dataset-Id` and `X-Detector-Id` headers to every API request so the backend resolves the correct per-dataset state.
- **State services** (`media-state`, `dataset-state`, `detector-state`, `vote-state`, `sort-state`, `settings-state`) hold client-side state and expose observables for reactive UI updates.
- **API services** (`medias-api`, `datasets-api`, `detectors-api`, etc.) wrap HTTP calls to the Flask backend. Each maps to a backend route module.

## Running unit tests

Karma has been removed from this project. There is currently no browser-based
frontend test runner configured. The Python backend tests (`./run-tests.sh core`)
include a frontend TypeScript build check to catch type errors.

## Code scaffolding

```bash
ng generate component component-name
```

For a complete list of available schematics (such as `components`, `directives`, or `pipes`), run:

```bash
ng generate --help
```
