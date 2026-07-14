import { EnvironmentProviders, Provider } from '@angular/core';
import { HttpInterceptorFn, provideHttpClient, withInterceptors } from '@angular/common/http';
import { provideHttpClientTesting } from '@angular/common/http/testing';

/**
 * Shared `TestBed` provider fragments for the frontend spec suite.
 *
 * Roughly sixty specs hand-copied `provideHttpClient() + provideHttpClientTesting()`
 * into their `configureTestingModule({ providers: [...] })`; a handful of
 * interceptor specs additionally wrap `provideHttpClient(withInterceptors([...]))`.
 * `provideHttpTesting()` is the single source of truth for that fragment — spread
 * it into the providers array so `HttpTestingController` is injectable.
 *
 * Sits next to `zoneless-testbed.ts` (`provideZoneless()`) and `mocks.ts` (shared
 * service stubs); the three compose freely in a `providers` array.
 */

/**
 * The HttpClient testing provider pair. Pass zero or more `HttpInterceptorFn`s to
 * register them under test (the interceptor specs' `withInterceptors([...])`
 * shape); with none, wires a plain testing client.
 *
 * ```ts
 * providers: [...provideZoneless(), ...provideHttpTesting()]
 * providers: [...provideHttpTesting(errorInterceptor)]  // interceptor under test
 * ```
 */
export function provideHttpTesting(
  ...interceptors: HttpInterceptorFn[]
): (Provider | EnvironmentProviders)[] {
  return [
    interceptors.length ? provideHttpClient(withInterceptors(interceptors)) : provideHttpClient(),
    provideHttpClientTesting(),
  ];
}
