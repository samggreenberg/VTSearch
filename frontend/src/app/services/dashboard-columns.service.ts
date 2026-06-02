import { Injectable } from '@angular/core';
import { ColMeta, ManagedColumns } from '../utils/managed-columns';

export type DatasetColumn =
  | 'name'
  | 'media_type'
  | 'num_items'
  | 'created_at'
  | 'expires_at'
  | 'created_by'
  | 'readers'
  | 'loaded'
  | 'actions';

export type DetectorColumn =
  | 'name'
  | 'media_type'
  | 'num_training'
  | 'autodetect'
  | 'last_trained_at'
  | 'created_at'
  | 'detector_loaded'
  | 'actions';

// `name` is pinned at the far left and `actions` at the far right; both are
// rendered explicitly by the Dashboard template, so they are excluded from the
// reorderable middle-column defaults. (Leaving `name` in here produced a
// duplicate Name header and shifted every body cell one column to the right,
// leaving the Actions column with no body cells.)
const DATASET_COLUMNS_DEFAULT: DatasetColumn[] = [
  'media_type',
  'num_items',
  'created_at',
  'expires_at',
  'created_by',
  'readers',
  'loaded',
];

const DETECTOR_COLUMNS_DEFAULT: DetectorColumn[] = [
  'media_type',
  'num_training',
  'autodetect',
  'last_trained_at',
  'created_at',
  'detector_loaded',
];

const DATASET_COL_ORDER_KEY = 'vtsearch.dashboard.datasetColumnOrder';
const DETECTOR_COL_ORDER_KEY = 'vtsearch.dashboard.detectorColumnOrder';

export const DATASET_COL_META: Record<string, ColMeta> = {
  name: { label: 'Name', title: 'Dataset display name (click to sort)', sortable: true },
  media_type: { label: 'Type', title: 'Media type: audio, image, text, video, or document (click to sort)', sortable: true },
  num_items: { label: '# Items', title: 'Number of media items in the dataset (click to sort)', sortable: true },
  created_at: { label: 'Created', title: 'When the dataset was first imported (click to sort)', sortable: true },
  expires_at: { label: 'Age-Off', title: 'When this dataset ages off and is automatically removed (click to sort)', sortable: true },
  created_by: { label: 'Creator', title: 'User who created this dataset (click to sort)', sortable: true },
  readers: { label: 'Readers', title: 'Users with access to this dataset (click to sort)', sortable: true },
  loaded: { label: 'Loaded?', title: 'Whether the dataset is currently loaded in memory', sortable: false },
  actions: { label: 'Actions', title: 'Available operations for this dataset', sortable: false },
};

export const DETECTOR_COL_META: Record<string, ColMeta> = {
  name: { label: 'Name', title: 'Detector display name (click to sort)', sortable: true },
  media_type: { label: 'Type', title: 'Media type this detector operates on (click to sort)', sortable: true },
  num_training: { label: '# Training', title: 'Number of labeled training examples (click to sort)', sortable: true },
  autodetect: { label: 'Autorun?', title: 'Include this detector in CLI autorun (click to sort)', sortable: true },
  last_trained_at: { label: 'Last Trained', title: 'When the detector was last trained (click to sort)', sortable: true },
  created_at: { label: 'Created', title: 'When the detector was created (click to sort)', sortable: true },
  detector_loaded: { label: 'Loaded?', title: "Whether the detector's inference data is cached in memory", sortable: false },
  actions: { label: 'Actions', title: 'Available operations for this detector', sortable: false },
};

/**
 * Singleton owner of the Dashboard's two `ManagedColumns` instances
 * (datasets, detectors). Lifting them out of `DashboardComponent` lets
 * the top-bar context pulldowns mirror the Dashboard's column sort
 * without coupling to the Dashboard component.
 */
@Injectable({ providedIn: 'root' })
export class DashboardColumnsService {
  readonly datasetCols: ManagedColumns<DatasetColumn>;
  readonly detectorCols: ManagedColumns<DetectorColumn>;

  constructor() {
    this.datasetCols = new ManagedColumns<DatasetColumn>(
      DATASET_COLUMNS_DEFAULT,
      DATASET_COL_META,
      { initialSort: 'name', storageKey: DATASET_COL_ORDER_KEY },
    );
    this.detectorCols = new ManagedColumns<DetectorColumn>(
      DETECTOR_COLUMNS_DEFAULT,
      DETECTOR_COL_META,
      { initialSort: 'name', storageKey: DETECTOR_COL_ORDER_KEY },
    );
  }
}
