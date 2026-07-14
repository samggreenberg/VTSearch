import { TestBed } from '@angular/core/testing';
import { firstValueFrom } from 'rxjs';
import { DashboardModalsService } from './dashboard-modals.service';
import { AutoDetectResultsData, DatasetRegistryEntry } from '../models/api.models';

describe('DashboardModalsService', () => {
  let service: DashboardModalsService;

  beforeEach(() => {
    TestBed.configureTestingModule({ providers: [DashboardModalsService] });
    service = TestBed.inject(DashboardModalsService);
  });

  it('every modal starts closed', () => {
    expect(service.combineDatasets.open).toBe(false);
    expect(service.combineDetectors.open).toBe(false);
    expect(service.export.open).toBe(false);
    expect(service.portableExport.open).toBe(false);
    expect(service.addLabels.open).toBe(false);
    expect(service.findResults.open).toBe(false);
    expect(service.stats.open).toBe(false);
    expect(service.detectorStats.open).toBe(false);
  });

  it('openCombineDatasets carries the dataset list; close resets it', () => {
    const datasets = [{ id: 'd1' }, { id: 'd2' }] as DatasetRegistryEntry[];
    service.openCombineDatasets(datasets);
    expect(service.combineDatasets).toEqual({ open: true, datasets });

    service.closeCombineDatasets();
    expect(service.combineDatasets).toEqual({ open: false, datasets: [] });
  });

  it('openCombineDetectors / closeCombineDetectors toggles open', () => {
    service.openCombineDetectors();
    expect(service.combineDetectors.open).toBe(true);
    service.closeCombineDetectors();
    expect(service.combineDetectors.open).toBe(false);
  });

  it('openExport carries the detector name; close clears it', () => {
    service.openExport('my-detector');
    expect(service.export).toEqual({ open: true, detectorName: 'my-detector' });
    service.closeExport();
    expect(service.export).toEqual({ open: false, detectorName: '' });
  });

  it('openPortableExport carries id + name; close clears both', () => {
    service.openPortableExport('id-1', 'Nice Name');
    expect(service.portableExport).toEqual({ open: true, detectorId: 'id-1', detectorName: 'Nice Name' });
    service.closePortableExport();
    expect(service.portableExport).toEqual({ open: false, detectorId: '', detectorName: '' });
  });

  it('openAddLabels carries id + name; close clears both', () => {
    service.openAddLabels('id-2', 'Labeller');
    expect(service.addLabels).toEqual({ open: true, detectorId: 'id-2', detectorName: 'Labeller' });
    service.closeAddLabels();
    expect(service.addLabels).toEqual({ open: false, detectorId: '', detectorName: '' });
  });

  it('openFindResults carries the results payload; close resets to an empty result', () => {
    const data: AutoDetectResultsData = { results: { det: { hits: [] } } };
    service.openFindResults(data);
    expect(service.findResults).toEqual({ open: true, data });
    service.closeFindResults();
    expect(service.findResults).toEqual({ open: false, data: { results: {} } });
  });

  it('openStats carries dataset id + name; close clears both', () => {
    service.openStats('ds-1', 'Dataset One');
    expect(service.stats).toEqual({ open: true, datasetId: 'ds-1', datasetName: 'Dataset One' });
    service.closeStats();
    expect(service.stats).toEqual({ open: false, datasetId: '', datasetName: '' });
  });

  it('openDetectorStats carries detector id + name; close clears both', () => {
    service.openDetectorStats('det-1', 'Detector One');
    expect(service.detectorStats).toEqual({ open: true, detectorId: 'det-1', detectorName: 'Detector One' });
    service.closeDetectorStats();
    expect(service.detectorStats).toEqual({ open: false, detectorId: '', detectorName: '' });
  });

  it('emits the new state to observers of the matching stream', async () => {
    service.openExport('streamed');
    const state = await firstValueFrom(service.export$);
    expect(state).toEqual({ open: true, detectorName: 'streamed' });
  });
});
