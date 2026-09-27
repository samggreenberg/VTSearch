/**
 * Red callouts drawn over a live page before it is photographed.
 *
 * Shared by the docs harness (`capture.ts`, from each manifest shot's
 * `annotations`) and the slide shooter (`slides/figs/src/shoot-ui-figs.mjs`).
 * Three kinds:
 *
 *   - `box`       — a red outline round the target, with an optional label pill
 *   - `highlight` — the same outline, with everything else dimmed
 *   - `step`      — a numbered marker: the outline plus a filled red disc
 *                   carrying `step` (1, 2, 3 …) on its top-left corner, and an
 *                   optional label pill beside the disc
 *
 * `step` exists for the click-by-click material (#4202): someone meeting the
 * interface for the first time cannot find "the + button" from a sentence, so
 * the picture says where, and in what order.
 *
 * A target is resolved in Node, with Playwright's locators, rather than in the
 * page: that is what lets a callout name "the dataset row called photos" rather
 * than only a CSS selector, and `photos` is a prefix of `photos-prod`, so a
 * substring match would box the wrong row. Forms a target can take:
 *
 *   '.btn-good'                                  first visible match
 *   { selector, hasText }                        …whose text contains hasText
 *   { selector, name }                           …whose `.name-cell` reads exactly name
 *   { x, y, w, h }                               a viewport box, as is
 *
 * A target that resolves to nothing is an error, not a silent omission: a
 * numbered picture missing its "3" is worse than no picture.
 */

const ACCENT = '#e8453c';
const LAYER_ID = '__shot_callouts';

/** Every visual size, in CSS px, so one knob scales them for a smaller slot. */
const SIZES = {
  stroke: 4,
  pad: 5,
  radius: 9,
  disc: 44,
  discFont: 27,
  labelFont: 18,
};

const escapeRe = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

/** The viewport box of *target* (see the module docstring), or null. */
export async function resolveBox(page, target) {
  if (target && typeof target === 'object' && 'x' in target) return target;
  const spec = typeof target === 'string' ? { selector: target } : target;
  let loc = page.locator(spec.selector);
  if (spec.hasText) loc = loc.filter({ hasText: spec.hasText });
  if (spec.name) {
    loc = loc.filter({
      has: page.locator('.name-cell', { hasText: new RegExp(`^\\s*${escapeRe(spec.name)}\\s*$`) }),
    });
  }
  const n = await loc.count();
  for (let i = 0; i < n; i++) {
    const b = await loc.nth(i).boundingBox();
    if (b && b.width > 0 && b.height > 0) return { x: b.x, y: b.y, w: b.width, h: b.height };
  }
  return null;
}

/**
 * Draw *callouts* over *page*. `scale` multiplies every size in `SIZES`.
 *
 * Replaces any layer an earlier call drew, so a harness can photograph the same
 * moment clean and then numbered without the first set leaking into the second.
 */
export async function drawCallouts(page, callouts, { scale = 1 } = {}) {
  const resolved = [];
  for (const c of callouts) {
    const box = await resolveBox(page, c.target);
    if (!box) throw new Error(`callout target not found: ${JSON.stringify(c.target)}`);
    resolved.push({ ...c, box });
  }
  const sizes = Object.fromEntries(Object.entries(SIZES).map(([k, v]) => [k, v * scale]));
  await page.evaluate(
    ({ items, sizes, accent, layerId }) => {
      document.getElementById(layerId)?.remove();
      const layer = document.createElement('div');
      layer.id = layerId;
      Object.assign(layer.style, {
        position: 'fixed', inset: '0', zIndex: '2147483647', pointerEvents: 'none',
      });
      document.body.appendChild(layer);
      const vw = window.innerWidth;
      const vh = window.innerHeight;
      const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
      const pill = (text, left, top) => {
        const l = document.createElement('div');
        l.textContent = text;
        Object.assign(l.style, {
          position: 'absolute', left: `${left}px`, top: `${top}px`,
          background: accent, color: '#fff',
          font: `600 ${sizes.labelFont}px system-ui, sans-serif`,
          padding: `${sizes.labelFont * 0.2}px ${sizes.labelFont * 0.6}px`,
          borderRadius: `${sizes.labelFont * 0.4}px`, whiteSpace: 'nowrap',
          boxShadow: '0 1px 3px rgba(0,0,0,0.35)',
        });
        layer.appendChild(l);
        // Keep the pill on screen: a callout near the right edge flips left.
        const r = l.getBoundingClientRect();
        if (r.right > vw - 4) l.style.left = `${Math.max(4, vw - 4 - r.width)}px`;
        return l;
      };
      for (const a of items) {
        const { x, y, w, h } = a.box;
        const pad = sizes.pad;
        const d = document.createElement('div');
        Object.assign(d.style, {
          position: 'absolute',
          left: `${x - pad}px`, top: `${y - pad}px`,
          width: `${w + pad * 2}px`, height: `${h + pad * 2}px`,
          border: `${sizes.stroke}px solid ${accent}`, borderRadius: `${sizes.radius}px`,
          boxShadow: a.kind === 'highlight' ? '0 0 0 4000px rgba(0,0,0,0.28)' : 'none',
          boxSizing: 'border-box',
        });
        layer.appendChild(d);
        if (a.kind === 'step') {
          // The disc straddles the outline's top-left corner, pulled back on
          // screen when the target hugs a viewport edge.
          const r = sizes.disc / 2;
          const cx = clamp(x - pad, r + 2, vw - r - 2);
          const cy = clamp(y - pad, r + 2, vh - r - 2);
          const disc = document.createElement('div');
          disc.textContent = String(a.step);
          Object.assign(disc.style, {
            position: 'absolute', left: `${cx - r}px`, top: `${cy - r}px`,
            width: `${sizes.disc}px`, height: `${sizes.disc}px`, borderRadius: '50%',
            background: accent, color: '#fff', border: `${sizes.stroke * 0.6}px solid #fff`,
            boxSizing: 'border-box', display: 'flex', alignItems: 'center', justifyContent: 'center',
            font: `700 ${sizes.discFont}px system-ui, sans-serif`, lineHeight: '1',
            boxShadow: '0 1px 4px rgba(0,0,0,0.4)',
          });
          layer.appendChild(disc);
          if (a.label) pill(a.label, cx + r + sizes.pad, cy - sizes.labelFont * 0.75);
        } else if (a.label) {
          const above = y > 60;
          pill(a.label, x - pad, above ? y - pad - sizes.labelFont * 2 : y + h + pad + 6);
        }
      }
    },
    { items: resolved, sizes, accent: ACCENT, layerId: LAYER_ID }
  );
}

/** Remove whatever `drawCallouts` drew. */
export async function clearCallouts(page) {
  await page.evaluate((id) => document.getElementById(id)?.remove(), LAYER_ID);
}
