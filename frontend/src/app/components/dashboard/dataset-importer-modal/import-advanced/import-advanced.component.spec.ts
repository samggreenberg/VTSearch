import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ImportAdvancedComponent } from './import-advanced.component';
import { provideZoneless } from '../../../../testing/zoneless-testbed';
import { settleZoneless } from '../../../../testing/settle-resource';
import {
  CleanerInfo,
  CleanerSelection,
  ClipperInfo,
  ClipperParameter,
  ConverterInfo,
  EmbedderInfo,
  SourceSpec,
} from '../../../../models/api.models';

describe('ImportAdvancedComponent', () => {
  let component: ImportAdvancedComponent;
  let fixture: ComponentFixture<ImportAdvancedComponent>;

  const embedders: EmbedderInfo[] = [
    { name: 'clap', display_name: 'CLAP', is_default: true, license_notice: null } as EmbedderInfo,
    { name: 'custom', display_name: 'Custom', is_default: false, license_notice: 'restricted' } as EmbedderInfo,
    { name: 'patchy', is_default: false, supports_patch_regions: true, license_notice: 'patch-license' } as EmbedderInfo,
    { name: 'struct', is_default: false, supports_geometric_verification: true } as EmbedderInfo,
  ];

  const clippers: ClipperInfo[] = [
    { name: 'clip_default', display_name: 'Default clipper' } as ClipperInfo,
    { name: 'sliding', display_name: 'Sliding window' } as ClipperInfo,
  ];

  const tolParam: ClipperParameter = { key: 'tol', label: 'Tolerance', type: 'number', default: 16 };

  const cleaners: CleanerInfo[] = [
    { name: 'exif', display_name: 'EXIF Orientation', media_type: 'image', default_enabled: true } as CleanerInfo,
    {
      name: 'trim',
      display_name: 'Solid Border Trim',
      media_type: 'image',
      default_enabled: false,
      parameters: [tolParam],
    } as CleanerInfo,
  ];

  const defaultSelection: CleanerSelection[] = [{ name: 'exif', params: {} }];

  const converters: ConverterInfo[] = [
    { name: 'video2image', source_type: 'video', target_type: 'image', fields: [] } as ConverterInfo,
  ];

  async function setInputs(inputs: Record<string, unknown>): Promise<void> {
    for (const [key, value] of Object.entries(inputs)) {
      fixture.componentRef.setInput(key, value);
    }
    await settleZoneless(fixture);
  }

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [ImportAdvancedComponent],
      providers: [...provideZoneless()],
    }).compileComponents();

    fixture = TestBed.createComponent(ImportAdvancedComponent);
    component = fixture.componentInstance;
  });

  it('creates', () => {
    expect(component).toBeTruthy();
  });

  describe('toggleAdvanced', () => {
    it('flips the advanced-open flag', () => {
      expect(component.advancedOpen).toBe(false);
      component.toggleAdvanced();
      expect(component.advancedOpen).toBe(true);
      component.toggleAdvanced();
      expect(component.advancedOpen).toBe(false);
    });
  });

  describe('isDefaultEmbedderSelected', () => {
    beforeEach(async () => {
      await setInputs({ embedders });
    });

    it('is true when no embedder is selected', async () => {
      await setInputs({ selectedEmbedder: '' });
      expect(component.isDefaultEmbedderSelected).toBe(true);
    });

    it('is true when the selected embedder is flagged is_default', async () => {
      await setInputs({ selectedEmbedder: 'clap' });
      expect(component.isDefaultEmbedderSelected).toBe(true);
    });

    it('is false when a non-default embedder is selected', async () => {
      await setInputs({ selectedEmbedder: 'custom' });
      expect(component.isDefaultEmbedderSelected).toBe(false);
    });
  });

  describe('isDefaultClipperSelected', () => {
    it('is true when the first clipper is selected', async () => {
      await setInputs({ clippers, selectedClipper: 'clip_default' });
      expect(component.isDefaultClipperSelected).toBe(true);
    });

    it('is false when a different clipper is selected', async () => {
      await setInputs({ clippers, selectedClipper: 'sliding' });
      expect(component.isDefaultClipperSelected).toBe(false);
    });

    it('is false when there are no clippers', async () => {
      await setInputs({ clippers: [], selectedClipper: '' });
      expect(component.isDefaultClipperSelected).toBe(false);
    });
  });

  describe('showAdvancedToggle', () => {
    it('is true whenever the source-specs block is offered', async () => {
      await setInputs({ showSourceSpecs: true, embedders, selectedEmbedder: 'custom' });
      expect(component.showAdvancedToggle).toBe(true);
    });

    it('is true when neither embedder nor clipper has been overridden', async () => {
      await setInputs({ showSourceSpecs: false, embedders, clippers, selectedEmbedder: 'clap', selectedClipper: 'clip_default' });
      expect(component.showAdvancedToggle).toBe(true);
    });

    it('is false when an override keeps a picker visible anyway', async () => {
      await setInputs({ showSourceSpecs: false, embedders, clippers, selectedEmbedder: 'custom', selectedClipper: 'clip_default' });
      expect(component.showAdvancedToggle).toBe(false);
    });
  });

  describe('showEmbedderPicker', () => {
    it('is hidden when a Solo embedder is locked', async () => {
      await setInputs({ embedders, lockedEmbedder: 'clap' });
      component.advancedOpen = true;
      expect(component.showEmbedderPicker).toBe(false);
    });

    it('is visible when Advanced is open', async () => {
      await setInputs({ embedders, selectedEmbedder: 'clap' });
      component.advancedOpen = true;
      expect(component.showEmbedderPicker).toBe(true);
    });

    it('is visible when a non-default embedder is chosen even while collapsed', async () => {
      await setInputs({ embedders, selectedEmbedder: 'custom' });
      component.advancedOpen = false;
      expect(component.showEmbedderPicker).toBe(true);
    });
  });

  describe('showClipperPicker', () => {
    it('is visible when Advanced is open', async () => {
      await setInputs({ clippers, selectedClipper: 'clip_default' });
      component.advancedOpen = true;
      expect(component.showClipperPicker).toBe(true);
    });

    it('is visible when a non-default clipper is chosen while collapsed', async () => {
      await setInputs({ clippers, selectedClipper: 'sliding' });
      component.advancedOpen = false;
      expect(component.showClipperPicker).toBe(true);
    });
  });

  describe('showStandaloneClipperButton', () => {
    it('is false when there is at most one clipper', async () => {
      await setInputs({ clippers: [clippers[0]], selectedClipper: 'clip_default' });
      component.advancedOpen = true;
      expect(component.showStandaloneClipperButton).toBe(false);
    });

    it('is false when the clipper picker is hidden', async () => {
      await setInputs({ clippers, selectedClipper: 'clip_default' });
      component.advancedOpen = false;
      expect(component.showStandaloneClipperButton).toBe(false);
    });

    it('is false when the source-specs column already hosts the native Details button', async () => {
      await setInputs({ clippers, selectedClipper: 'clip_default', showSourceSpecs: true, availableConverters: converters });
      component.advancedOpen = true;
      expect(component.showStandaloneClipperButton).toBe(false);
    });

    it('is true when the picker is visible and no source-specs column is present', async () => {
      await setInputs({ clippers, selectedClipper: 'clip_default', showSourceSpecs: false });
      component.advancedOpen = true;
      expect(component.showStandaloneClipperButton).toBe(true);
    });

    it('is true when the source-specs flow is on but no converters feed it (empty column suppressed)', async () => {
      // showSourceSpecs is true but there are no non-native converters, so the
      // Include-media column collapses to a trivial native-only row and is
      // hidden; the clipper chooser must stay reachable via the standalone button.
      await setInputs({ clippers, selectedClipper: 'clip_default', showSourceSpecs: true, availableConverters: [] });
      component.advancedOpen = true;
      expect(component.showStandaloneClipperButton).toBe(true);
    });
  });

  describe('hasSourceSpecsColumn', () => {
    it('is false when the source-specs flow is off', async () => {
      await setInputs({ showSourceSpecs: false, availableConverters: converters });
      expect(component.hasSourceSpecsColumn).toBe(false);
    });

    it('is false when there are no non-native converters', async () => {
      await setInputs({ showSourceSpecs: true, availableConverters: [] });
      expect(component.hasSourceSpecsColumn).toBe(false);
    });

    it('is true when the flow is on and at least one converter feeds the native type', async () => {
      await setInputs({ showSourceSpecs: true, availableConverters: converters });
      expect(component.hasSourceSpecsColumn).toBe(true);
    });
  });

  describe('embedder option grouping', () => {
    beforeEach(async () => {
      await setInputs({ embedders });
    });

    it('recommendedEmbedders holds the is_default embedders', () => {
      expect(component.recommendedEmbedders.map((e) => e.name)).toEqual(['clap']);
    });

    it('advancedEmbedderOptions holds the non-default embedders', () => {
      expect(component.advancedEmbedderOptions.map((e) => e.name)).toEqual(['custom', 'patchy', 'struct']);
    });

    it('embedderLabel prefers the display name and falls back to the name', () => {
      expect(component.embedderLabel(embedders[0])).toBe('CLAP');
      expect(component.embedderLabel(embedders[2])).toBe('patchy');
    });

    it('patchEmbedderOptions holds patch-capable embedders', () => {
      expect(component.patchEmbedderOptions.map((e) => e.name)).toEqual(['patchy']);
    });

    it('structuralEmbedderOptions holds geometric-verification embedders', () => {
      expect(component.structuralEmbedderOptions.map((e) => e.name)).toEqual(['struct']);
    });
  });

  describe('license notices', () => {
    beforeEach(async () => {
      await setInputs({ embedders });
    });

    it('licenseNotice reflects the selected primary embedder', async () => {
      await setInputs({ selectedEmbedder: 'custom' });
      expect(component.licenseNotice).toBe('restricted');
    });

    it('licenseNotice is null when the embedder has no notice', async () => {
      await setInputs({ selectedEmbedder: 'clap' });
      expect(component.licenseNotice).toBeNull();
    });

    it('licenseNotice is null when nothing is selected', () => {
      expect(component.licenseNotice).toBeNull();
    });

    it('patchLicenseNotice reflects the selected patch embedder', async () => {
      await setInputs({ selectedPatchEmbedder: 'patchy' });
      expect(component.patchLicenseNotice).toBe('patch-license');
    });

    it('structuralLicenseNotice is null for an embedder without a notice', async () => {
      await setInputs({ selectedStructuralEmbedder: 'struct' });
      expect(component.structuralLicenseNotice).toBeNull();
    });
  });

  describe('clipperDisplayName', () => {
    beforeEach(async () => {
      await setInputs({ clippers });
    });

    it('is None when no clipper matches the selection', async () => {
      await setInputs({ selectedClipper: '' });
      expect(component.clipperDisplayName).toBe('None');
    });

    it('is None for a *_default clipper', async () => {
      await setInputs({ selectedClipper: 'clip_default' });
      expect(component.clipperDisplayName).toBe('None');
    });

    it('is the display name for a real clipper', async () => {
      await setInputs({ selectedClipper: 'sliding' });
      expect(component.clipperDisplayName).toBe('Sliding window');
    });
  });

  describe('output emitters', () => {
    it('onSourceSpecsChange re-emits the specs list', () => {
      const specs: SourceSpec[] = [{ source_type: 'image', converter: null, params: {} }];
      let emitted: SourceSpec[] | null = null;
      component.sourceSpecsChange.subscribe((s: SourceSpec[]) => (emitted = s));
      component.onSourceSpecsChange(specs);
      expect(emitted).toBe(specs);
    });

    it('onEmbedderChange emits the selected embedder name', () => {
      let emitted = '';
      component.selectedEmbedderChange.subscribe((n: string) => (emitted = n));
      component.onEmbedderChange('custom');
      expect(emitted).toBe('custom');
    });

    it('onPatchEmbedderChange emits the patch embedder name', () => {
      let emitted = '';
      component.selectedPatchEmbedderChange.subscribe((n: string) => (emitted = n));
      component.onPatchEmbedderChange('patchy');
      expect(emitted).toBe('patchy');
    });

    it('onStructuralEmbedderChange emits the structural embedder name', () => {
      let emitted = '';
      component.selectedStructuralEmbedderChange.subscribe((n: string) => (emitted = n));
      component.onStructuralEmbedderChange('struct');
      expect(emitted).toBe('struct');
    });

    it('onClipperClick requests the parent chooser', () => {
      let fired = false;
      component.clipperChooserRequested.subscribe(() => (fired = true));
      component.onClipperClick();
      expect(fired).toBe(true);
    });

    it('onBuildProjectionChange emits the toggle value', () => {
      let emitted: boolean | null = null;
      component.buildProjectionChange.subscribe((v: boolean) => (emitted = v));
      component.onBuildProjectionChange(true);
      expect(emitted).toBe(true);
    });

    it('onMergeNearDuplicatesChange emits the toggle value', () => {
      let emitted: boolean | null = null;
      component.mergeNearDuplicatesChange.subscribe((v: boolean) => (emitted = v));
      component.onMergeNearDuplicatesChange(true);
      expect(emitted).toBe(true);
    });
  });
  describe('cleanup gates', () => {
    it('showCleanupSection stays hidden when no cleaners are registered', async () => {
      await setInputs({ cleaners: [], selectedCleaners: [] });
      component.advancedOpen = true;
      expect(component.showCleanupSection).toBe(false);
    });

    it('showCleanupSection follows the Advanced toggle at the registry default', async () => {
      await setInputs({ cleaners, selectedCleaners: defaultSelection });
      expect(component.isDefaultCleanupSelected).toBe(true);
      expect(component.showCleanupSection).toBe(false);
      component.advancedOpen = true;
      expect(component.showCleanupSection).toBe(true);
    });

    it('a non-default selection keeps the block visible with Advanced collapsed', async () => {
      await setInputs({ cleaners, selectedCleaners: [] });
      expect(component.isDefaultCleanupSelected).toBe(false);
      expect(component.showCleanupSection).toBe(true);
    });

    it('a parameter override counts as non-default', async () => {
      await setInputs({ cleaners, selectedCleaners: [{ name: 'exif', params: { tol: 8 } }] });
      expect(component.isDefaultCleanupSelected).toBe(false);
    });

    it('a non-default cleanup selection also keeps the Advanced toggle visible', async () => {
      await setInputs({ cleaners, selectedCleaners: [], embedders, clippers, selectedClipper: 'clip_default' });
      expect(component.showAdvancedToggle).toBe(false);
      await setInputs({ selectedCleaners: defaultSelection });
      expect(component.showAdvancedToggle).toBe(true);
    });

    it('isCleanerEnabled reflects the selection', async () => {
      await setInputs({ cleaners, selectedCleaners: defaultSelection });
      expect(component.isCleanerEnabled('exif')).toBe(true);
      expect(component.isCleanerEnabled('trim')).toBe(false);
    });

    it('onCleanerToggle adds a cleaner in registry order', async () => {
      await setInputs({ cleaners, selectedCleaners: [{ name: 'trim', params: {} }] });
      let emitted: CleanerSelection[] = [];
      component.selectedCleanersChange.subscribe((v: CleanerSelection[]) => (emitted = v));
      component.onCleanerToggle(cleaners[0], true);
      expect(emitted.map((c) => c.name)).toEqual(['exif', 'trim']);
    });

    it('onCleanerToggle removes a cleaner', async () => {
      await setInputs({ cleaners, selectedCleaners: defaultSelection });
      let emitted: CleanerSelection[] | null = null;
      component.selectedCleanersChange.subscribe((v: CleanerSelection[]) => (emitted = v));
      component.onCleanerToggle(cleaners[0], false);
      expect(emitted).toEqual([]);
    });

    it('cleanerParamValue falls back to the descriptor default', async () => {
      await setInputs({ cleaners, selectedCleaners: [{ name: 'trim', params: {} }] });
      expect(component.cleanerParamValue(cleaners[1], tolParam)).toBe(16);
      await setInputs({ selectedCleaners: [{ name: 'trim', params: { tol: 4 } }] });
      expect(component.cleanerParamValue(cleaners[1], tolParam)).toBe(4);
    });

    it('onCleanerParamChange records an override and drops a default-valued one', async () => {
      await setInputs({ cleaners, selectedCleaners: [{ name: 'trim', params: {} }] });
      let emitted: CleanerSelection[] = [];
      component.selectedCleanersChange.subscribe((v: CleanerSelection[]) => (emitted = v));

      component.onCleanerParamChange(cleaners[1], tolParam, 4);
      expect(emitted[0].params).toEqual({ tol: 4 });

      await setInputs({ selectedCleaners: [{ name: 'trim', params: { tol: 4 } }] });
      component.onCleanerParamChange(cleaners[1], tolParam, 16);
      expect(emitted[0].params).toEqual({});
    });

    it('renders one checkbox per cleaner, checked per the selection', async () => {
      component.advancedOpen = true;
      await setInputs({ cleaners, selectedCleaners: defaultSelection });
      const boxes = fixture.nativeElement.querySelectorAll('.cleanup-row input[type="checkbox"]');
      expect(boxes.length).toBe(2);
      expect((boxes[0] as HTMLInputElement).checked).toBe(true);
      expect((boxes[1] as HTMLInputElement).checked).toBe(false);
    });

    it('renders parameter inputs only for a checked cleaner that declares them', async () => {
      component.advancedOpen = true;
      await setInputs({ cleaners, selectedCleaners: defaultSelection });
      expect(fixture.nativeElement.querySelectorAll('.cleanup-param').length).toBe(0);

      await setInputs({ selectedCleaners: [{ name: 'trim', params: {} }] });
      expect(fixture.nativeElement.querySelectorAll('.cleanup-param').length).toBe(1);
    });
  });
});
