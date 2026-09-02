import { apiErrorMessage } from './api-error';

describe('apiErrorMessage', () => {
  it('reads the message key of the standard envelope', () => {
    expect(
      apiErrorMessage({ error: { code: 400, status: 'Bad Request', message: 'Missing field' } }, 'fallback'),
    ).toBe('Missing field');
  });

  it('reads a global handler envelope carrying request_id and extras', () => {
    expect(
      apiErrorMessage(
        { error: { code: 409, status: 'Conflict', message: 'Dataset is not loaded', request_id: 'r1', error_code: 'dataset_not_loaded' } },
        'fallback',
      ),
    ).toBe('Dataset is not loaded');
  });

  it('falls back for empty, missing, or non-string payloads', () => {
    expect(apiErrorMessage({ error: { message: '' } }, 'fb')).toBe('fb');
    expect(apiErrorMessage({ error: {} }, 'fb')).toBe('fb');
    expect(apiErrorMessage({}, 'fb')).toBe('fb');
    expect(apiErrorMessage(null, 'fb')).toBe('fb');
    expect(apiErrorMessage(undefined, 'fb')).toBe('fb');
    expect(apiErrorMessage({ error: { errors: { field: ['bad'] } } }, 'fb')).toBe('fb');
    expect(apiErrorMessage({ error: { message: 42 } }, 'fb')).toBe('fb');
    // ProgressEvent bodies (network failure) have no keys at all.
    expect(apiErrorMessage({ error: new ProgressEvent('error') }, 'fb')).toBe('fb');
  });

  it('no longer reads the retired {error: ...} spelling', () => {
    // Regression guard for the envelope unification: if a route ever
    // reintroduces `{"error": ...}` the fallback fires loudly rather than
    // the helper silently propping up a second envelope.
    expect(apiErrorMessage({ error: { error: 'No files uploaded' } }, 'fb')).toBe('fb');
  });
});
