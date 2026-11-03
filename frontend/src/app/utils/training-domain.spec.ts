import { domainShiftLine, domainShiftWarning } from './training-domain';

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

  it('warns on a likely shift, naming the detector and the dataset', () => {
    expect(domainShiftWarning(report(true), 'Birds', 'screens')).toEqual({
      message: '31% of this dataset looks atypical vs photos: likely domain shift',
      detail:
        'Birds learned from photos (40 of its 52 labelled examples are there). ' +
        'Check its calls on screens by hand before relying on them.',
    });
  });

  it('says nothing in-domain, or with no report', () => {
    expect(domainShiftWarning(report(false), 'Birds', 'screens')).toBeNull();
    // The schema allows a null report (nothing to compare); the generated type does not say so.
    const refused = { ...report(true), domain_shift: null } as unknown as Parameters<typeof domainShiftWarning>[0];
    expect(domainShiftWarning(refused, 'Birds', 'screens')).toBeNull();
    expect(domainShiftWarning(null, 'Birds', 'screens')).toBeNull();
  });

  it('falls back to the reference id when it has no name', () => {
    expect(domainShiftWarning({ ...report(true), reference_dataset_name: '' }, 'Birds', 'screens')?.message).toContain(
      'vs ref-id',
    );
  });
});
