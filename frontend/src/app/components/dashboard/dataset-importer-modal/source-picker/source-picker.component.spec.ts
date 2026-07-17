import { ComponentFixture, TestBed } from '@angular/core/testing';

import { SourcePickerComponent } from './source-picker.component';
import { provideZoneless } from '../../../../testing/zoneless-testbed';
import { settleZoneless } from '../../../../testing/settle-resource';
import { ManagedColumns } from '../../../../utils/managed-columns';
import { DemoDatasetEntry } from '../../../../generated/api-client/models/demo-dataset-entry';
import { ImporterInfo, ImporterPickerTab, MediaTypeInfo } from '../../../../models/api.models';

describe('SourcePickerComponent', () => {
  let component: SourcePickerComponent;
  let fixture: ComponentFixture<SourcePickerComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [SourcePickerComponent],
      providers: [...provideZoneless()],
    }).compileComponents();

    fixture = TestBed.createComponent(SourcePickerComponent);
    component = fixture.componentInstance;
  });

  it('creates', () => {
    expect(component).toBeTruthy();
  });

  describe('chrome hint templates', () => {
    it('noImporterHint substitutes {label} with the active tab label', async () => {
      fixture.componentRef.setInput('noImporterHintTemplate', 'Pick how to add a {label} dataset.');
      fixture.componentRef.setInput('activeImporterTabLabel', 'Audio');
      await settleZoneless(fixture);
      expect(component.noImporterHint).toBe('Pick how to add a Audio dataset.');
    });

    it('tabTitle substitutes {label} with the passed label', async () => {
      fixture.componentRef.setInput('tabTitleTemplate', 'Show {label} importers');
      await settleZoneless(fixture);
      expect(component.tabTitle('Local')).toBe('Show Local importers');
    });
  });

  describe('demo media-type helpers', () => {
    const mediaTypes: MediaTypeInfo[] = [
      { type_id: 'audio', name: 'Audio', icon: 'audio-icon' },
      { type_id: 'image', name: 'Image', icon: '' },
    ];

    beforeEach(async () => {
      fixture.componentRef.setInput('mediaTypes', mediaTypes);
      await settleZoneless(fixture);
    });

    it('getDemoTabIcon returns the matching media type icon', () => {
      expect(component.getDemoTabIcon('audio')).toBe('audio-icon');
    });

    it('getDemoTabIcon returns empty string when the type has no icon', () => {
      expect(component.getDemoTabIcon('image')).toBe('');
    });

    it('getDemoTabIcon returns empty string for an unknown media type', () => {
      expect(component.getDemoTabIcon('video')).toBe('');
    });

    it('getDemoTabText returns the human name for a known type', () => {
      expect(component.getDemoTabText('audio')).toBe('Audio');
    });

    it('getDemoTabText falls back to the raw id for an unknown type', () => {
      expect(component.getDemoTabText('video')).toBe('video');
    });
  });

  describe('demo row helpers', () => {
    const demo = { name: 'gtzan', label: 'GTZAN', status: 'ready' } as DemoDatasetEntry;

    it('demoRowDisabled returns false when no predicate is provided', () => {
      expect(component.demoRowDisabled(demo)).toBe(false);
    });

    it('demoRowDisabled delegates to the provided predicate', async () => {
      fixture.componentRef.setInput('demoRowDisabledFn', (d: DemoDatasetEntry) => d.name === 'gtzan');
      await settleZoneless(fixture);
      expect(component.demoRowDisabled(demo)).toBe(true);
      expect(component.demoRowDisabled({ name: 'other' } as DemoDatasetEntry)).toBe(false);
    });

    it('demoRowTitle returns empty string when no formatter is provided', () => {
      expect(component.demoRowTitle(demo)).toBe('');
    });

    it('demoRowTitle delegates to the provided formatter', async () => {
      fixture.componentRef.setInput('demoRowTitleFn', (d: DemoDatasetEntry) => `title:${d.name}`);
      await settleZoneless(fixture);
      expect(component.demoRowTitle(demo)).toBe('title:gtzan');
    });
  });

  describe('status badge helpers', () => {
    it('maps status to the badge class', () => {
      expect(component.statusBadgeClass('ready')).toBe('badge-ready');
      expect(component.statusBadgeClass('needs_embedding')).toBe('badge-embedding');
      expect(component.statusBadgeClass('needs_download')).toBe('badge-download');
    });

    it('maps status to the badge label', () => {
      expect(component.statusBadgeLabel('ready')).toBe('Ready');
      expect(component.statusBadgeLabel('needs_embedding')).toBe('Needs setup');
      expect(component.statusBadgeLabel('needs_download')).toBe('Needs Download');
    });
  });

  describe('onDemoHeaderClick', () => {
    function makeCols(): ManagedColumns {
      return new ManagedColumns(
        ['label', 'num_files'],
        {
          label: { label: 'Name', title: '', sortable: true },
          num_files: { label: 'Files', title: '', sortable: false },
        },
        { initialSort: 'label' },
      );
    }

    it('sorts when the clicked column is sortable', () => {
      const cols = makeCols();
      fixture.componentRef.setInput('demoCols', cols);
      const spy = vi.spyOn(cols, 'sortBy');
      component.onDemoHeaderClick('label');
      expect(spy).toHaveBeenCalledWith('label');
    });

    it('does nothing when the clicked column is not sortable', () => {
      const cols = makeCols();
      fixture.componentRef.setInput('demoCols', cols);
      const spy = vi.spyOn(cols, 'sortBy');
      component.onDemoHeaderClick('num_files');
      expect(spy).not.toHaveBeenCalled();
    });

    it('is a no-op when demoCols is null', () => {
      fixture.componentRef.setInput('demoCols', null);
      expect(() => component.onDemoHeaderClick('label')).not.toThrow();
    });
  });

  describe('tab bar rendering + outputs', () => {
    const tabs: ImporterPickerTab[] = [
      { id: 'local', label: 'Local' },
      { id: 'server', label: 'Server' },
    ];
    const importers: ImporterInfo[] = [
      { name: 'server_folder', display_name: 'Server folder', description: 'From a path' },
      { name: 'server_zip', display_name: 'Server zip' },
    ];

    it('renders one button per visible tab and emits activeTabChange on click', async () => {
      fixture.componentRef.setInput('visibleImporterTabs', tabs);
      await settleZoneless(fixture);

      const buttons = fixture.nativeElement.querySelectorAll('.tab-bar .tab');
      expect(buttons.length).toBe(2);

      let emitted = '';
      component.activeTabChange.subscribe((id: string) => (emitted = id));
      buttons[1].click();
      expect(emitted).toBe('server');
    });

    it('shows the no-tab hint when no tab is active', async () => {
      fixture.componentRef.setInput('visibleImporterTabs', tabs);
      fixture.componentRef.setInput('noTabHint', 'Choose a category.');
      await settleZoneless(fixture);
      expect(fixture.nativeElement.querySelector('.tab-bar-hint').textContent).toContain('Choose a category.');
    });

    it('renders sub-tabs for the active category and emits importerSelected on click', async () => {
      fixture.componentRef.setInput('visibleImporterTabs', tabs);
      fixture.componentRef.setInput('activeTab', 'server');
      fixture.componentRef.setInput('importersForActiveTab', importers);
      await settleZoneless(fixture);

      const subtabs = fixture.nativeElement.querySelectorAll('.importer-subtab');
      expect(subtabs.length).toBe(2);

      let picked: ImporterInfo | null = null;
      component.importerSelected.subscribe((imp: ImporterInfo) => (picked = imp));
      subtabs[0].click();
      expect(picked).toBe(importers[0]);
    });

    it('shows the empty-category text when the active tab has no importers', async () => {
      fixture.componentRef.setInput('visibleImporterTabs', tabs);
      fixture.componentRef.setInput('activeTab', 'server');
      fixture.componentRef.setInput('emptyCategoryText', 'Nothing here.');
      fixture.componentRef.setInput('importersForActiveTab', []);
      await settleZoneless(fixture);
      expect(fixture.nativeElement.querySelector('.importer-subtab-bar').textContent).toContain('Nothing here.');
    });

    it('hides the lone sub-tab by default but shows it when alwaysShowSubtabBar is set', async () => {
      fixture.componentRef.setInput('visibleImporterTabs', tabs);
      fixture.componentRef.setInput('activeTab', 'server');
      fixture.componentRef.setInput('importersForActiveTab', [importers[0]]);
      await settleZoneless(fixture);
      expect(fixture.nativeElement.querySelector('.importer-subtab')).toBeNull();

      fixture.componentRef.setInput('alwaysShowSubtabBar', true);
      await settleZoneless(fixture);
      expect(fixture.nativeElement.querySelector('.importer-subtab')).not.toBeNull();
    });
  });
});
