import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { CombineDatasetsModalComponent } from './combine-datasets-modal.component';
import { DatasetRegistryEntry } from '../../../models/api.models';

describe('CombineDatasetsModalComponent', () => {
  let component: CombineDatasetsModalComponent;
  let fixture: ComponentFixture<CombineDatasetsModalComponent>;
  let httpMock: HttpTestingController;

  const mockMediaTypes = {
    media_types: [
      { type_id: 'audio', name: 'Audio', icon: '🎵' },
      { type_id: 'image', name: 'Image', icon: '🖼' },
    ],
  };

  const ds = (id: string, name: string, type: string, num: number, pkl: string): DatasetRegistryEntry => ({
    id,
    name,
    media_type: type,
    num_items: num,
    pkl_path: pkl,
  });

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [CombineDatasetsModalComponent],
      providers: [provideHttpClient(), provideHttpClientTesting()],
    }).compileComponents();

    fixture = TestBed.createComponent(CombineDatasetsModalComponent);
    component = fixture.componentInstance;
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => {
    httpMock.verify();
  });

  function flushMediaTypes() {
    httpMock.expectOne('/api/media-types').flush(mockMediaTypes);
  }

  it('builds rows from input datasets, filtering out entries with no pkl_path', () => {
    component.datasets = [
      ds('a', 'Alpha', 'audio', 10, '/data/a.pkl'),
      ds('b', 'Bravo', 'audio', 20, ''),
      ds('c', 'Charlie', 'audio', 30, '/data/c.pkl'),
    ];
    fixture.detectChanges();
    flushMediaTypes();

    expect(component.rows.length).toBe(2);
    expect(component.rows.map((r) => r.id)).toEqual(['a', 'c']);
    expect(component.totalItems).toBe(40);
  });

  it('canCombine is true only when ≥2 rows share a single media type', () => {
    component.datasets = [
      ds('a', 'Alpha', 'audio', 10, '/x/a.pkl'),
      ds('b', 'Bravo', 'audio', 20, '/x/b.pkl'),
    ];
    fixture.detectChanges();
    flushMediaTypes();

    expect(component.canCombine).toBeTrue();
    expect(component.disabledReason).toBe('');
  });

  it('canCombine is false when media types differ', () => {
    component.datasets = [
      ds('a', 'Alpha', 'audio', 10, '/x/a.pkl'),
      ds('b', 'Bravo', 'image', 20, '/x/b.pkl'),
    ];
    fixture.detectChanges();
    flushMediaTypes();

    expect(component.canCombine).toBeFalse();
    expect(component.disabledReason).toContain('same media type');
  });

  it('canCombine is false when fewer than two rows remain after removal', () => {
    component.datasets = [
      ds('a', 'Alpha', 'audio', 10, '/x/a.pkl'),
      ds('b', 'Bravo', 'audio', 20, '/x/b.pkl'),
    ];
    fixture.detectChanges();
    flushMediaTypes();

    component.removeRow('b');
    expect(component.rows.length).toBe(1);
    expect(component.canCombine).toBeFalse();
    expect(component.disabledReason).toContain('at least two');
  });

  it('submit posts the pkl paths to /api/dataset/combine and emits combineStarted', () => {
    component.datasets = [
      ds('a', 'Alpha', 'audio', 10, '/x/a.pkl'),
      ds('b', 'Bravo', 'audio', 20, '/x/b.pkl'),
    ];
    fixture.detectChanges();
    flushMediaTypes();

    let started = false;
    component.combineStarted.subscribe(() => { started = true; });

    component.submit();
    const req = httpMock.expectOne('/api/dataset/combine');
    expect(req.request.method).toBe('POST');
    expect(req.request.body).toEqual({ datasets: ['/x/a.pkl', '/x/b.pkl'] });
    req.flush({ ok: true });

    expect(started).toBeTrue();
    expect(component.submitting).toBeFalse();
  });

  it('submit surfaces backend errors without emitting combineStarted', () => {
    component.datasets = [
      ds('a', 'Alpha', 'audio', 10, '/x/a.pkl'),
      ds('b', 'Bravo', 'audio', 20, '/x/b.pkl'),
    ];
    fixture.detectChanges();
    flushMediaTypes();

    let started = false;
    component.combineStarted.subscribe(() => { started = true; });

    component.submit();
    const req = httpMock.expectOne('/api/dataset/combine');
    req.flush({ error: 'boom' }, { status: 500, statusText: 'Server Error' });

    expect(started).toBeFalse();
    expect(component.error).toBe('boom');
    expect(component.submitting).toBeFalse();
  });

  it('submit is a no-op when canCombine is false', () => {
    component.datasets = [ds('a', 'Alpha', 'audio', 10, '/x/a.pkl')];
    fixture.detectChanges();
    flushMediaTypes();

    component.submit();
    httpMock.expectNone('/api/dataset/combine');
  });
});
