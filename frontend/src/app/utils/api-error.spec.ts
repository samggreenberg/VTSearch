import { apiErrorMessage } from './api-error';

describe('apiErrorMessage', () => {
  it('reads the error_response envelope ({error: ...})', () => {
    expect(apiErrorMessage({ error: { error: 'No files uploaded' } }, 'fallback')).toBe(
      'No files uploaded',
    );
  });

  it('reads the flask_smorest envelope ({message: ...})', () => {
    // Regression: inline handlers used to read only err.error?.error, so
    // every abort(400, message=...) route degraded to the generic fallback.
    expect(
      apiErrorMessage({ error: { code: 400, status: 'Bad Request', message: 'Missing field' } }, 'fallback'),
    ).toBe('Missing field');
  });

  it('prefers the error key when both are present', () => {
    expect(apiErrorMessage({ error: { error: 'specific', message: 'generic' } }, 'fb')).toBe(
      'specific',
    );
  });

  it('falls back for empty, missing, or non-string payloads', () => {
    expect(apiErrorMessage({ error: { error: '' } }, 'fb')).toBe('fb');
    expect(apiErrorMessage({ error: {} }, 'fb')).toBe('fb');
    expect(apiErrorMessage({}, 'fb')).toBe('fb');
    expect(apiErrorMessage(null, 'fb')).toBe('fb');
    expect(apiErrorMessage(undefined, 'fb')).toBe('fb');
    expect(apiErrorMessage({ error: { errors: { field: ['bad'] } } }, 'fb')).toBe('fb');
    expect(apiErrorMessage({ error: { message: 42 } }, 'fb')).toBe('fb');
    // ProgressEvent bodies (network failure) have no keys at all.
    expect(apiErrorMessage({ error: new ProgressEvent('error') }, 'fb')).toBe('fb');
  });
});
