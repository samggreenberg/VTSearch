import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ImportConfigComponent } from './import-config.component';
import { provideZoneless } from '../../../../testing/zoneless-testbed';
import { settleZoneless } from '../../../../testing/settle-resource';

describe('ImportConfigComponent', () => {
  let component: ImportConfigComponent;
  let fixture: ComponentFixture<ImportConfigComponent>;

  const options = ['audio', 'image', 'text'];

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [ImportConfigComponent],
      providers: [...provideZoneless()],
    }).compileComponents();

    fixture = TestBed.createComponent(ImportConfigComponent);
    component = fixture.componentInstance;
    fixture.componentRef.setInput('mediaTypeOptions', options);
    await settleZoneless(fixture);
  });

  it('creates', () => {
    expect(component).toBeTruthy();
  });

  describe('id derivation', () => {
    it('derives listboxId from the trigger field id', async () => {
      fixture.componentRef.setInput('mediaTypeFieldId', 'sf-media-type');
      await settleZoneless(fixture);
      expect(component.listboxId).toBe('sf-media-type-listbox');
    });

    it('derives per-option ids from the trigger field id', async () => {
      fixture.componentRef.setInput('mediaTypeFieldId', 'sf-media-type');
      await settleZoneless(fixture);
      expect(component.optionId(2)).toBe('sf-media-type-option-2');
    });

    it('activeDescendantId is null while closed', () => {
      expect(component.activeDescendantId).toBeNull();
    });

    it('activeDescendantId points at the active option once open', () => {
      component.toggle();
      expect(component.open).toBe(true);
      expect(component.activeDescendantId).toBe(component.optionId(component.activeIndex));
    });
  });

  describe('option label + icon lookup', () => {
    it('optionLabel returns the mapped label', async () => {
      fixture.componentRef.setInput('mediaTypeOptionLabels', { audio: 'Audio files' });
      await settleZoneless(fixture);
      expect(component.optionLabel('audio')).toBe('Audio files');
    });

    it('optionLabel falls back to the raw value when unmapped', () => {
      expect(component.optionLabel('audio')).toBe('audio');
    });

    it('iconFor returns the mapped icon or empty string', async () => {
      fixture.componentRef.setInput('mediaTypeOptionIcons', { audio: 'audio-icon' });
      await settleZoneless(fixture);
      expect(component.iconFor('audio')).toBe('audio-icon');
      expect(component.iconFor('image')).toBe('');
    });
  });

  describe('open/close behaviour', () => {
    it('toggle opens the popup and seeds the highlight on the current selection', () => {
      component.mediaType = 'image';
      component.toggle();
      expect(component.open).toBe(true);
      expect(component.activeIndex).toBe(options.indexOf('image'));
    });

    it('toggle seeds the highlight on the first option when nothing is selected', () => {
      component.mediaType = '';
      component.toggle();
      expect(component.activeIndex).toBe(0);
    });

    it('toggle a second time closes the popup and clears the highlight', () => {
      component.toggle();
      component.toggle();
      expect(component.open).toBe(false);
      expect(component.activeIndex).toBe(-1);
    });
  });

  describe('select', () => {
    it('emits mediaTypeChange and closes when a new value is chosen', () => {
      component.mediaType = 'audio';
      component.open = true;
      let emitted = '';
      component.mediaTypeChange.subscribe((v: string) => (emitted = v));
      component.select('image');
      expect(emitted).toBe('image');
      expect(component.open).toBe(false);
    });

    it('does not emit when the current value is re-selected', () => {
      component.mediaType = 'audio';
      let emitted = false;
      component.mediaTypeChange.subscribe(() => (emitted = true));
      component.select('audio');
      expect(emitted).toBe(false);
    });
  });

  describe('keyboard handling', () => {
    function key(k: string): KeyboardEvent {
      return new KeyboardEvent('keydown', { key: k });
    }

    it('opens the popup on ArrowDown while closed', () => {
      component.onTriggerKeydown(key('ArrowDown'));
      expect(component.open).toBe(true);
    });

    it('ignores other keys while closed', () => {
      component.onTriggerKeydown(key('a'));
      expect(component.open).toBe(false);
    });

    it('moves the highlight down and up, clamped to bounds', () => {
      component.toggle();
      component.activeIndex = 0;
      component.onTriggerKeydown(key('ArrowDown'));
      expect(component.activeIndex).toBe(1);
      component.onTriggerKeydown(key('ArrowUp'));
      expect(component.activeIndex).toBe(0);
      component.onTriggerKeydown(key('ArrowUp'));
      expect(component.activeIndex).toBe(0);
    });

    it('Home and End jump to the first and last option', () => {
      component.toggle();
      component.onTriggerKeydown(key('End'));
      expect(component.activeIndex).toBe(options.length - 1);
      component.onTriggerKeydown(key('Home'));
      expect(component.activeIndex).toBe(0);
    });

    it('Enter commits the highlighted option', () => {
      component.mediaType = 'audio';
      component.toggle();
      component.activeIndex = 1;
      let emitted = '';
      component.mediaTypeChange.subscribe((v: string) => (emitted = v));
      component.onTriggerKeydown(key('Enter'));
      expect(emitted).toBe('image');
      expect(component.open).toBe(false);
    });

    it('Escape closes the popup without emitting', () => {
      component.toggle();
      let emitted = false;
      component.mediaTypeChange.subscribe(() => (emitted = true));
      component.onTriggerKeydown(key('Escape'));
      expect(component.open).toBe(false);
      expect(emitted).toBe(false);
    });

    it('Tab closes the popup', () => {
      component.toggle();
      component.onTriggerKeydown(key('Tab'));
      expect(component.open).toBe(false);
    });
  });

  describe('outside-click dismissal', () => {
    it('closes when the click is outside the host element', () => {
      component.open = true;
      const outside = document.createElement('div');
      document.body.appendChild(outside);
      component.onDocumentClick({ target: outside } as unknown as MouseEvent);
      expect(component.open).toBe(false);
      outside.remove();
    });

    it('stays open when the click is inside the host element', () => {
      component.open = true;
      const inside = fixture.nativeElement as HTMLElement;
      component.onDocumentClick({ target: inside } as unknown as MouseEvent);
      expect(component.open).toBe(true);
    });

    it('is a no-op when already closed', () => {
      component.open = false;
      component.onDocumentClick({ target: document.body } as unknown as MouseEvent);
      expect(component.open).toBe(false);
    });
  });
});
