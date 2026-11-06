import { DomainShiftSurface, domainShiftLine, domainShiftWarning } from './training-domain';

/** The training-domain warning's words, shared by Tune's toast, Find Results and a run's notice (#4961, #5501). */
describe('training-domain warning', () => {
  const report = (shifted: boolean) => ({
    reference_dataset_id: 'ref-id',
    reference_dataset_name: 'photos',
    labels_total: 52,
    labels_in_reference: 40,
    labels_in_active: 0,
    domain_shift: {
      reference_dataset_id: 'ref-id',
      n_items: 4000,
      alpha: 0.05,
      frac_atypical: 0.314,
      expected_atypical: 0.05,
      z_score: 75,
      median_pvalue: 0.18,
      shifted,
    },
    reason: '',
  });

  it('words the line off the figure', () => {
    expect(domainShiftLine(0.685, 'photos')).toBe('69% of this dataset looks atypical vs photos: likely domain shift');
  });

  it('warns on a likely shift, quoting the detector and the datasets (#5545)', () => {
    expect(domainShiftWarning(report(true), 'Mammal detector', 'Caltech-101 (M)', 'toast')).toEqual({
      message: '31% of this dataset looks atypical vs photos: likely domain shift',
      detail: '"Mammal detector" learned mostly from "photos". Tune it to check its calls on "Caltech-101 (M)".',
    });
  });

  it('says how much it learned there as a word, and nothing when it learned it all there (#5545)', () => {
    const detail = (inReference: number) =>
      domainShiftWarning({ ...report(true), labels_in_reference: inReference }, 'Birds', 'screens', 'toast')?.detail;
    expect(detail(52)).toMatch(/^"Birds" learned from "photos"\. /);
    expect(detail(27)).toMatch(/^"Birds" learned mostly from "photos"\. /);
    // Half is not most: the reference holds the most of them, not always a majority.
    expect(detail(26)).toMatch(/^"Birds" learned partly from "photos"\. /);
    expect(detail(10)).toMatch(/^"Birds" learned partly from "photos"\. /);
  });

  it('points each surface at the control it has for the next step (#5545)', () => {
    const next = (surface: DomainShiftSurface) =>
      domainShiftWarning(report(true), 'Birds', 'screens', surface)?.detail.replace(/^.*?"photos"\. /, '');
    expect(next('toast')).toBe('Tune it to check its calls on "screens".');
    expect(next('tune')).toBe('Check its calls on "screens" here, on the Autopilot tab.');
    expect(next('results')).toBe('Check its matches with Browse, or test it in Tune.');
  });

  it('says this detector and this dataset, unquoted, when it has no names for them', () => {
    expect(domainShiftWarning(report(true), undefined, null, 'toast')?.detail).toBe(
      'This detector learned mostly from "photos". Tune it to check its calls on this dataset.',
    );
  });

  it('says nothing in-domain, or with no report', () => {
    expect(domainShiftWarning(report(false), 'Birds', 'screens', 'toast')).toBeNull();
    // The schema allows a null report (nothing to compare); the generated type does not say so.
    const refused = { ...report(true), domain_shift: null } as unknown as Parameters<typeof domainShiftWarning>[0];
    expect(domainShiftWarning(refused, 'Birds', 'screens', 'toast')).toBeNull();
    expect(domainShiftWarning(null, 'Birds', 'screens', 'toast')).toBeNull();
  });

  it('falls back to the reference id when it has no name', () => {
    const warning = domainShiftWarning({ ...report(true), reference_dataset_name: '' }, 'Birds', 'screens', 'toast');
    expect(warning?.message).toContain('vs ref-id');
    expect(warning?.detail).toContain('from "ref-id".');
  });
});
