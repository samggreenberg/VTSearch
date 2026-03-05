import { ChartsService } from './charts.service';

describe('ChartsService', () => {
  let service: ChartsService;

  beforeEach(() => {
    service = new ChartsService();
  });

  function createMockCanvas(): HTMLCanvasElement {
    const canvas = document.createElement('canvas');
    canvas.width = 600;
    canvas.height = 300;
    return canvas;
  }

  it('should be created', () => {
    expect(service).toBeTruthy();
  });

  describe('renderErrorCostChart', () => {
    it('should handle empty data', () => {
      const canvas = createMockCanvas();
      expect(() => service.renderErrorCostChart(canvas, [])).not.toThrow();
    });

    it('should render with valid data', () => {
      const canvas = createMockCanvas();
      const data = [
        { num_labels: 5, error_cost: 0.8 },
        { num_labels: 10, error_cost: 0.5 },
        { num_labels: 15, error_cost: 0.3 },
      ];
      expect(() => service.renderErrorCostChart(canvas, data)).not.toThrow();
    });

    it('should handle single data point', () => {
      const canvas = createMockCanvas();
      const data = [{ num_labels: 5, error_cost: 0.5 }];
      expect(() => service.renderErrorCostChart(canvas, data)).not.toThrow();
    });
  });

  describe('renderStabilityChart', () => {
    it('should handle empty data', () => {
      const canvas = createMockCanvas();
      expect(() => service.renderStabilityChart(canvas, [])).not.toThrow();
    });

    it('should render with valid data', () => {
      const canvas = createMockCanvas();
      const data = [
        { num_labels: 5, num_flips: 3 },
        { num_labels: 10, num_flips: 1 },
      ];
      expect(() => service.renderStabilityChart(canvas, data)).not.toThrow();
    });
  });

  describe('renderDiversityChart', () => {
    it('should handle empty data', () => {
      const canvas = createMockCanvas();
      expect(() => service.renderDiversityChart(canvas, [])).not.toThrow();
    });

    it('should render with valid data', () => {
      const canvas = createMockCanvas();
      const data = [
        { num_labels: 5, diversity_level: 2, depth: 5 },
        { num_labels: 10, diversity_level: 3, depth: 5 },
      ];
      expect(() => service.renderDiversityChart(canvas, data)).not.toThrow();
    });
  });
});
