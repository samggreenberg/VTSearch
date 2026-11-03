import { AUTO_EXPORT_FIX_HINT, autoExportFailure } from './auto-export-status';

describe('autoExportFailure', () => {
  const failed = { exporter: 'webhook', exporter_display_name: 'Webhook (HTTP POST)', success: false };

  it('names the exporter by its display name and says where to fix it (#5463)', () => {
    const text = autoExportFailure({ ...failed, error: 'Connection refused' });
    expect(text).toBe(
      'Sending the results via Webhook (HTTP POST) failed: Connection refused. Check the exporter in Settings › AutoFind.',
    );
    expect(text).not.toContain('via webhook');
    expect(text).toContain(AUTO_EXPORT_FIX_HINT);
  });

  it('does not double the stop after a reason that already ends a sentence', () => {
    expect(autoExportFailure({ ...failed, error: 'The URL is invalid.' })).toContain('invalid. Check');
  });

  it('says so when the server gave no reason', () => {
    expect(autoExportFailure(failed)).toContain('failed: unknown error. Check');
  });
});
