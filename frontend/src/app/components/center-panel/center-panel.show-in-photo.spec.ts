import { ComponentFixture, TestBed } from '@angular/core/testing';
import { HttpTestingController } from '@angular/common/http/testing';
import { provideRouter } from '@angular/router';
import { signal } from '@angular/core';
import { Observable, of } from 'rxjs';
import { vi } from 'vitest';

import { CenterPanelComponent } from './center-panel.component';
import { Media } from '../../models/api.models';
import { ActiveContextService } from '../../services/active-context.service';
import { SourceLookup, SourcePhotoRequest, SourcePhotoService } from '../../services/source-photo.service';
import { configureZoneless } from '../../testing/zoneless-testbed';
import { settleZoneless } from '../../testing/settle-resource';
import { provideHttpTesting } from '../../testing/test-providers';

/**
 * Show in photo's button on the focused item (#4750): offered only on a
 * dataset whose import produced a photo sibling, and only for an item whose
 * own lookup does not come back saying it has no source.
 */
describe('CenterPanelComponent Show in photo', () => {
  let fixture: ComponentFixture<CenterPanelComponent>;
  let httpMock: HttpTestingController;
  let answer: SourceLookup;
  let fake: {
    offered: ReturnType<typeof signal<boolean>>;
    request: ReturnType<typeof signal<SourcePhotoRequest | null>>;
    lookup: ReturnType<typeof vi.fn<(id: number) => Observable<SourceLookup>>>;
    open: ReturnType<typeof vi.fn<(id: number) => void>>;
    close: ReturnType<typeof vi.fn<() => void>>;
  };

  // Document media keeps the viewer inert in jsdom (see the zoneless canary).
  const face: Media = { id: 4, media_type: 'document', filename: 'a.jpg→face_0.jpg', md5: 'm', custom_metadata: {} };

  beforeEach(async () => {
    answer = { kind: 'found', datasetId: 'photos', mediaId: 2, box: [0.1, 0.2, 0.3, 0.4] };
    fake = {
      offered: signal(true),
      request: signal<SourcePhotoRequest | null>(null),
      lookup: vi.fn(() => of(answer)),
      open: vi.fn(),
      close: vi.fn(),
    };
    configureZoneless({
      imports: [CenterPanelComponent],
      providers: [...provideHttpTesting(), provideRouter([]), { provide: SourcePhotoService, useValue: fake }],
    });
    TestBed.inject(ActiveContextService).setActivePair('faces', 'det');
    httpMock = TestBed.inject(HttpTestingController);
    fixture = TestBed.createComponent(CenterPanelComponent);
  });

  afterEach(() => {
    fixture.destroy();
    httpMock.verify();
  });

  async function show(media: Media): Promise<void> {
    fixture.componentRef.setInput('media', media);
    await settleZoneless(fixture);
  }

  function button(): HTMLButtonElement | null {
    const all = Array.from(fixture.nativeElement.querySelectorAll('button')) as HTMLButtonElement[];
    return all.find((b) => b.textContent?.trim() === 'Show in photo') ?? null;
  }

  it('offers it for an item whose photo was found, and opens that item', async () => {
    await show(face);
    expect(fake.lookup).toHaveBeenCalledWith(4);
    button()!.click();
    expect(fake.open).toHaveBeenCalledWith(4);
  });

  it('offers it when the photo dataset is not loaded: the overlay loads it', async () => {
    answer = { kind: 'not-loaded', datasetId: 'photos' };
    await show(face);
    expect(button()).not.toBeNull();
  });

  it('withdraws it for an item with no source', async () => {
    answer = { kind: 'none', message: 'This item was not made by a converter.' };
    await show(face);
    expect(button()).toBeNull();
  });

  it('neither offers it nor asks on a dataset with no photo sibling', async () => {
    fake.offered.set(false);
    await show(face);
    expect(button()).toBeNull();
    expect(fake.lookup).not.toHaveBeenCalled();
  });

  it('does not ask again when the same item is re-hydrated', async () => {
    await show(face);
    await show({ ...face, custom_metadata: { Source: '/tmp/a.jpg' } });
    expect(fake.lookup).toHaveBeenCalledTimes(1);
  });

  it('closes an open overlay when the pane goes away', async () => {
    await show(face);
    fixture.destroy();
    expect(fake.close).toHaveBeenCalled();
  });
});
