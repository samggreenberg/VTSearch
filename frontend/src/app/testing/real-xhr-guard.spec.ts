import { TestBed } from '@angular/core/testing';
import { HttpClient } from '@angular/common/http';
import { firstValueFrom } from 'rxjs';
import { installRealXhrGuard, takeRealXhrRequests } from './real-xhr-guard';

// `src/test-setup.ts` installs the guard and fails any test that leaves a
// recorded request behind, so each spec here drains what it provokes.
describe('real XHR guard (#4808)', () => {
  it('stops a real open() before the network and records it', () => {
    const xhr = new XMLHttpRequest();
    expect(() => xhr.open('POST', '/api/medias/batch')).toThrow(/provideHttpTesting/);

    const requests = takeRealXhrRequests();
    expect(requests).toHaveLength(1);
    expect(requests[0]).toContain('POST /api/medias/batch');
    expect(requests[0]).toContain('stops a real open() before the network');
    expect(takeRealXhrRequests()).toEqual([]);
  });

  it('catches the root HttpClient a spec gets without provideHttpTesting()', async () => {
    const http = TestBed.inject(HttpClient);
    await expect(firstValueFrom(http.get('/api/medias/ids'))).rejects.toThrow(/provideHttpTesting/);
    expect(takeRealXhrRequests().map((r) => r.split(' (')[0])).toEqual(['GET /api/medias/ids']);
  });

  it('wraps open() only once however often the setup file re-runs', () => {
    const open = XMLHttpRequest.prototype.open;
    installRealXhrGuard();
    expect(XMLHttpRequest.prototype.open).toBe(open);
  });
});
