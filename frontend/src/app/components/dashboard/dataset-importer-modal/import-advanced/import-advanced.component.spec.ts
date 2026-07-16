import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ImportAdvancedComponent } from './import-advanced.component';
import { provideZoneless } from '../../../../testing/zoneless-testbed';
import { settleZoneless } from '../../../../testing/settle-resource';
import { ClipperInfo, EmbedderInfo, SourceSpec } from '../../../../models/api.models';

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
      await setInputs({ clippers, selectedClipper: 'clip_default', showSourceSpecs: true });
      component.advancedOpen = true;
      expect(component.showStandaloneClipperButton).toBe(false);
    });

    it('is true when the picker is visible and no source-specs column is present', async () => {
      await setInputs({ clippers, selectedClipper: 'clip_default', showSourceSpecs: false });
      component.advancedOpen = true;
      expect(component.showStandaloneClipperButton).toBe(true);
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
});
