#!/usr/bin/env node
/**
 * Cached wrapper around `ng-openapi-gen`.
 *
 * `ng-openapi-gen` regenerates the TypeScript API client from `openapi.json` on
 * every `pretest` / `prebuild` hook, even when the spec hasn't moved. On a cold
 * `run-tests.sh` the frontend gate pays that cost on every invocation for no
 * benefit. This wrapper stamps a hash of the inputs (the OpenAPI spec, the
 * `ng-openapi-gen.json` config, and the generator's own version) and skips the
 * run when the stamp still matches AND the output directory is present.
 *
 * The stamp lives at `src/app/generated/.openapi-gen.hash` — inside the
 * gitignored `src/app/generated/` tree but OUTSIDE the generator's `output`
 * directory, so `removeStaleFiles: true` never wipes it.
 *
 * Pass `--force` (or set `OPENAPI_GEN_FORCE=1`) to regenerate unconditionally;
 * that is what the explicit `generate-api-client` script does.
 */
import { createHash } from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { readFileSync, existsSync, mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const frontendRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const configPath = join(frontendRoot, 'ng-openapi-gen.json');

/** Read a file as UTF-8, or return null if it is missing/unreadable. */
function readOrNull(path) {
  try {
    return readFileSync(path, 'utf8');
  } catch {
    return null;
  }
}

const configText = readOrNull(configPath);
if (configText === null) {
  console.error(`openapi-gen-cached: cannot read ${configPath}`);
  process.exit(1);
}
const config = JSON.parse(configText);
const specPath = resolve(frontendRoot, config.input ?? 'openapi.json');
const outputDir = resolve(frontendRoot, config.output ?? 'src/app/generated/api-client');
const stampPath = join(frontendRoot, 'src', 'app', 'generated', '.openapi-gen.hash');

const specText = readOrNull(specPath);
if (specText === null) {
  console.error(`openapi-gen-cached: cannot read OpenAPI spec at ${specPath}`);
  process.exit(1);
}

// A generator upgrade can change the emitted client even when the spec is
// byte-identical, so fold its version into the cache key.
const genVersion =
  JSON.parse(readOrNull(join(frontendRoot, 'node_modules', 'ng-openapi-gen', 'package.json')) ?? '{}')
    .version ?? 'unknown';

const hash = createHash('sha256');
hash.update('v1\0'); // stamp-format tag; bump to invalidate every cache at once
hash.update(genVersion);
hash.update('\0');
hash.update(configText);
hash.update('\0');
hash.update(specText);
const inputHash = hash.digest('hex');

const force = process.argv.includes('--force') || process.env.OPENAPI_GEN_FORCE === '1';

if (!force && existsSync(outputDir) && readOrNull(stampPath) === inputHash) {
  console.log('openapi-gen-cached: API client up to date, skipping ng-openapi-gen.');
  process.exit(0);
}

// Regenerate, then stamp only on success.
const bin = join(
  frontendRoot,
  'node_modules',
  '.bin',
  process.platform === 'win32' ? 'ng-openapi-gen.cmd' : 'ng-openapi-gen',
);
const result = spawnSync(bin, [], {
  stdio: 'inherit',
  cwd: frontendRoot,
  shell: process.platform === 'win32',
});
if (result.status !== 0) {
  process.exit(result.status ?? 1);
}

mkdirSync(dirname(stampPath), { recursive: true });
writeFileSync(stampPath, inputHash);
