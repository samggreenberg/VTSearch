# VTSearch Frontend

Angular SPA for the VTSearch media explorer. Built with Angular CLI 19.2 and TypeScript.

## Development server

Start the Flask backend first (`python app.py --local` from the project root), then:

```bash
npm start
```

This starts the Angular dev server at `http://localhost:4200/` with a proxy that forwards `/api/*` requests to the Flask backend at `localhost:5000`. The application automatically reloads when source files change.

## Building for production

```bash
npm run build:prod
```

This compiles the Angular app and outputs the build artifacts to `../static/` (the project root's `static/` directory), where Flask serves them. Output files: `index.html`, `main.js`, `polyfills.js`, `styles.css`.

You must run `npm install` before the first build to install Angular CLI and other dependencies locally.

## Running unit tests

```bash
ng test --watch=false
```

Uses [Karma](https://karma-runner.github.io) as the test runner.

## Code scaffolding

```bash
ng generate component component-name
```

For a complete list of available schematics (such as `components`, `directives`, or `pipes`), run:

```bash
ng generate --help
```
