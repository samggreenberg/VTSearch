import { ComponentFixture, TestBed } from '@angular/core/testing';
import { toUsageBytes, UsageBarComponent, UsageBytes } from './usage-bar.component';
import { provideZoneless } from '../../../testing/zoneless-testbed';

const GB = 1024 ** 3;

describe('UsageBarComponent', () => {
  let component: UsageBarComponent;
  let fixture: ComponentFixture<UsageBarComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [UsageBarComponent],
      providers: [...provideZoneless()],
    }).compileComponents();
    fixture = TestBed.createComponent(UsageBarComponent);
    component = fixture.componentInstance;
  });

  function withUsage(usage: UsageBytes): UsageBarComponent {
    fixture.componentRef.setInput('usage', usage);
    return component;
  }

  it('counts the free space in datasets the size of the largest', () => {
    fixture.componentRef.setInput('titlePrefix', 'Server disk');
    withUsage({ total: 500 * GB, used: 490 * GB, free: 10 * GB, datasetBytes: 3 * GB, datasetBytesSource: 'largest' });
    expect(component.headroomText).toBe('room for 3 more datasets the size of the largest (3.0 GB)');
    expect(component.title).toBe(
      'Server disk: 10.0 GB free of 500 GB, room for 3 more datasets the size of the largest (3.0 GB)',
    );
  });

  it('says when not even one more dataset fits', () => {
    withUsage({ total: 16 * GB, used: 15 * GB, free: GB, datasetBytes: 2 * GB, datasetBytesSource: 'largest' });
    expect(component.headroomText).toBe('no room for another dataset the size of the largest (2.0 GB)');
  });

  it('names the stand-in size as typical when no dataset is registered', () => {
    withUsage({ total: 16 * GB, used: 15 * GB, free: GB, datasetBytes: GB, datasetBytesSource: 'default' });
    expect(component.headroomText).toBe('room for 1 typical 1.0 GB dataset');
  });

  it('leaves the headroom out of the title when the server sent no dataset size', () => {
    fixture.componentRef.setInput('label', 'RAM');
    withUsage({ total: 16 * GB, used: 8 * GB, free: 8 * GB });
    expect(component.headroomText).toBe('');
    expect(component.title).toBe('RAM: 8.0 GB free of 16.0 GB');
  });

  it('maps a usage probe response onto the bar input', () => {
    expect(
      toUsageBytes({ total: 10, used: 4, free: 6, dataset_bytes: 2, dataset_bytes_source: 'largest', low: true }),
    ).toEqual({ total: 10, used: 4, free: 6, datasetBytes: 2, datasetBytesSource: 'largest', low: true });
  });
});
