/**
 * Red callouts drawn over a live page before it is photographed.
 *
 * Shared by the docs harness (`capture.ts`, from each manifest shot's
 * `annotations`) and the slide shooter (`slides/figs/src/shoot-ui-figs.mjs`).
 * Three kinds:
 *
 *   - `box`       — a red outline round the target, with an optional label pill
 *                   above it (or `at: 'right'` / `'bottom'`)
 *   - `highlight` — the same outline, with everything else dimmed
 *   - `step`      — a numbered marker: the outline plus a filled red disc
 *                   carrying `step` (1, 2, 3 …), and an optional label pill
 *                   beside the disc. `at` puts the disc against the target's
 *                   `left` edge (the default), `right`, `top` or `bottom` edge,
 *                   or on its top-left `corner`. Beside rather than on top: a
 *                   disc on the corner sits on whatever labels the control, and
 *                   those labels are half of what the picture is showing.
 *
 * `step` exists for the click-by-click material (#4202): someone meeting the
 * interface for the first time cannot find "the + button" from a sentence, so
 * the picture says where, and in what order.
 *
 * A target is resolved in Node, with Playwright's locators, rather than in the
 * page: that is what lets a callout name "the dataset row called photos-train"
 * rather than only a CSS selector, and `photos` is a prefix of both of the
 * deck's piles, so a substring match can box the wrong row. Forms a target can take:
 *
 *   '.btn-good'                                  first visible match
 *   { selector, hasText }                        …whose text contains hasText
 *   { selector, name }                           …whose `.name-cell` reads exactly name
 *   { x, y, w, h }                               a viewport box, as is
 *
 * A target that resolves to nothing is an error, not a silent omission: a
 * numbered picture missing its "3" is worse than no picture.
 *
 * Every mark stays inside the picture (#4686). An outline is drawn `pad` outside
 * its target, so the outline round a target that runs to the window's edge (a
 * whole side panel) lost that side, and a disc set beside a target near a
 * cropped shot's edge lost half its number. So outlines, discs and labels are
 * all kept inside the viewport (an outline is drawn just inside an edge its
 * target touches), `drawCallouts` returns the box its marks cover so a harness
 * that crops can grow the crop to hold them, and a mark that still does not fit
 * is an error.
 */

const ACCENT = '#e8453c';
const LAYER_ID = '__shot_callouts';

/** Every visual size, in CSS px, so one knob scales them for a smaller slot. */
const SIZES = {
  stroke: 4,
  pad: 5,
  radius: 9,
  disc: 38,
  discFont: 23,
  labelFont: 18,
  /** How far inside the viewport an outline whose target meets its edge is drawn. */
  edge: 2,
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
 *
 * Returns the viewport box `{ x, y, w, h }` every mark drawn lies within, so a
 * harness that crops the frame can widen the crop to take them all in.
 */
export async function drawCallouts(page, callouts, { scale = 1 } = {}) {
  const resolved = [];
  for (const c of callouts) {
    const box = await resolveBox(page, c.target);
    if (!box) throw new Error(`callout target not found: ${JSON.stringify(c.target)}`);
    resolved.push({ ...c, box });
  }
  const sizes = Object.fromEntries(Object.entries(SIZES).map(([k, v]) => [k, v * scale]));
  return page.evaluate(
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
      // Put a label pill at (left, top), pulled back on screen: a callout near
      // an edge slides along it rather than off it.
      const place = (l, left, top) => {
        const r = l.getBoundingClientRect();
        l.style.left = `${clamp(left, 4, vw - 4 - r.width)}px`;
        l.style.top = `${clamp(top, 4, vh - 4 - r.height)}px`;
        return l;
      };
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
        return place(l, left, top);
      };
      for (const [i, a] of items.entries()) {
        const { x, y, w, h } = a.box;
        const pad = sizes.pad;
        // The outline, `pad` clear of the target, except on a side where that
        // would put it past the viewport's edge (a panel running to the edge of
        // the window): there it is drawn just inside.
        const left = Math.max(x - pad, sizes.edge);
        const top = Math.max(y - pad, sizes.edge);
        const right = Math.min(x + w + pad, vw - sizes.edge);
        const bottom = Math.min(y + h + pad, vh - sizes.edge);
        if (right - left < sizes.stroke * 2 || bottom - top < sizes.stroke * 2) {
          throw new Error(`callout ${i + 1} (${a.label ?? a.step ?? a.kind}): its target is off screen`);
        }
        const d = document.createElement('div');
        Object.assign(d.style, {
          position: 'absolute',
          left: `${left}px`, top: `${top}px`,
          width: `${right - left}px`, height: `${bottom - top}px`,
          border: `${sizes.stroke}px solid ${accent}`, borderRadius: `${sizes.radius}px`,
          boxShadow: a.kind === 'highlight' ? '0 0 0 4000px rgba(0,0,0,0.28)' : 'none',
          boxSizing: 'border-box',
        });
        layer.appendChild(d);
        if (a.kind === 'step') {
          // The disc touches the outline on the side `at` names, pulled back
          // on screen when the target hugs a viewport edge.
          const r = sizes.disc / 2;
          const touch = r - sizes.stroke;
          const [px, py] = {
            left: [left - touch, (top + bottom) / 2],
            right: [right + touch, (top + bottom) / 2],
            top: [(left + right) / 2, top - touch],
            bottom: [(left + right) / 2, bottom + touch],
            corner: [left, top],
          }[a.at || 'left'];
          const cx = clamp(px, r + 2, vw - r - 2);
          const cy = clamp(py, r + 2, vh - r - 2);
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
          if (a.label) {
            // On the far side of the disc from the target, so it covers neither.
            const l = pill(a.label, cx + r + sizes.pad, cy - sizes.labelFont * 0.75);
            if (a.at === 'left' || !a.at) {
              place(l, cx - r - sizes.pad - l.getBoundingClientRect().width, cy - sizes.labelFont * 0.75);
            }
          }
        } else if (a.label) {
          // Above the box by default (below it when there is no room above);
          // `at: 'right'` beside it, for targets stacked too tight to label
          // from above without covering the one before.
          const mid = y + h / 2 - sizes.labelFont * 0.75;
          const tall = sizes.labelFont * 2;
          if (a.at === 'right') pill(a.label, x + w + pad + 8, mid);
          else if (a.at !== 'bottom' && y > 60) pill(a.label, x - pad, y - pad - tall);
          // No room above: below the box — or, for a target that runs to the
          // bottom of the viewport (a whole panel), inside the top of it.
          else if (y + h + pad + 6 + tall <= vh) pill(a.label, x - pad, y + h + pad + 6);
          else pill(a.label, x + pad + 4, y + pad + 6);
        }
      }
      // Everything above is placed to stay on screen. A mark that still is not
      // (a label wider than the window) would ship cropped, so it fails the
      // shot instead.
      let bounds = null;
      for (const el of layer.children) {
        const b = el.getBoundingClientRect();
        if (b.left < -0.5 || b.top < -0.5 || b.right > vw + 0.5 || b.bottom > vh + 0.5) {
          throw new Error(`callout mark "${(el.textContent || 'outline').slice(0, 40)}" runs off screen`);
        }
        bounds = bounds
          ? {
              left: Math.min(bounds.left, b.left), top: Math.min(bounds.top, b.top),
              right: Math.max(bounds.right, b.right), bottom: Math.max(bounds.bottom, b.bottom),
            }
          : { left: b.left, top: b.top, right: b.right, bottom: b.bottom };
      }
      return bounds && {
        x: bounds.left, y: bounds.top, w: bounds.right - bounds.left, h: bounds.bottom - bounds.top,
      };
    },
    { items: resolved, sizes, accent: ACCENT, layerId: LAYER_ID }
  );
}

/** Remove whatever `drawCallouts` drew. */
export async function clearCallouts(page) {
  await page.evaluate((id) => document.getElementById(id)?.remove(), LAYER_ID);
}
