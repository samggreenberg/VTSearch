import { Injectable } from '@angular/core';
import {
  ErrorCostDataPoint,
  StabilityDataPoint,
  DiversityDataPoint,
} from '../models/api.models';

interface ChartPadding {
  top: number;
  right: number;
  bottom: number;
  left: number;
}

@Injectable({ providedIn: 'root' })
export class ChartsService {
  private readonly padding: ChartPadding = { top: 20, right: 20, bottom: 40, left: 50 };

  private themeColor(varName: string): string {
    return getComputedStyle(document.documentElement).getPropertyValue(varName).trim();
  }

  private drawAxes(ctx: CanvasRenderingContext2D, width: number, height: number): void {
    const { top, left } = this.padding;
    const chartWidth = width - left - this.padding.right;
    const chartHeight = height - top - this.padding.bottom;

    ctx.strokeStyle = this.themeColor('--border');
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(left, top);
    ctx.lineTo(left, top + chartHeight);
    ctx.lineTo(left + chartWidth, top + chartHeight);
    ctx.stroke();
  }

  private drawGrid(ctx: CanvasRenderingContext2D, width: number, height: number): void {
    const { top, left } = this.padding;
    const chartWidth = width - left - this.padding.right;
    const chartHeight = height - top - this.padding.bottom;

    ctx.strokeStyle = this.themeColor('--border-subtle');
    ctx.lineWidth = 1;
    for (let i = 1; i <= 5; i++) {
      const y = top + (chartHeight * i) / 5;
      ctx.beginPath();
      ctx.moveTo(left, y);
      ctx.lineTo(left + chartWidth, y);
      ctx.stroke();
    }
  }

  private drawLine(
    ctx: CanvasRenderingContext2D,
    xs: number[],
    ys: number[],
    color: string,
  ): void {
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < xs.length; i++) {
      if (i === 0) ctx.moveTo(xs[i], ys[i]);
      else ctx.lineTo(xs[i], ys[i]);
    }
    ctx.stroke();

    ctx.fillStyle = color;
    for (let i = 0; i < xs.length; i++) {
      ctx.beginPath();
      ctx.arc(xs[i], ys[i], 3, 0, 2 * Math.PI);
      ctx.fill();
    }
  }

  private showEmpty(ctx: CanvasRenderingContext2D, width: number, height: number): void {
    ctx.fillStyle = this.themeColor('--text-muted');
    ctx.font = '14px sans-serif';
    ctx.fillText('No data available', 20, height / 2);
  }

  renderErrorCostChart(canvas: HTMLCanvasElement, data: ErrorCostDataPoint[]): void {
    const ctx = canvas.getContext('2d')!;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!data || data.length === 0) {
      this.showEmpty(ctx, canvas.width, canvas.height);
      return;
    }

    const { top, left } = this.padding;
    const chartWidth = canvas.width - left - this.padding.right;
    const chartHeight = canvas.height - top - this.padding.bottom;

    const numLabels = data.map((d) => d.num_labels);
    const errorCosts = data.map((d) => d.error_cost);
    const maxLabels = Math.max(...numLabels);
    const maxCost = Math.max(...errorCosts);
    const minCost = Math.min(...errorCosts);

    const xScale = (val: number) => left + (val / maxLabels) * chartWidth;
    const yScale = (val: number) =>
      top + chartHeight - ((val - minCost) / (maxCost - minCost || 1)) * chartHeight;

    this.drawAxes(ctx, canvas.width, canvas.height);
    this.drawGrid(ctx, canvas.width, canvas.height);

    const xs = numLabels.map(xScale);
    const ys = errorCosts.map(yScale);
    this.drawLine(ctx, xs, ys, this.themeColor('--accent'));

    ctx.fillStyle = this.themeColor('--text-secondary');
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Number of Labels', canvas.width / 2, canvas.height - 10);

    ctx.save();
    ctx.translate(15, canvas.height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Error Cost', 0, 0);
    ctx.restore();

    ctx.textAlign = 'center';
    ctx.fillText('0', left, canvas.height - this.padding.bottom + 15);
    ctx.fillText(maxLabels.toString(), left + chartWidth, canvas.height - this.padding.bottom + 15);
    ctx.textAlign = 'right';
    ctx.fillText(maxCost.toFixed(2), left - 5, top + 5);
    ctx.fillText(minCost.toFixed(2), left - 5, top + chartHeight + 5);
  }

  renderStabilityChart(canvas: HTMLCanvasElement, data: StabilityDataPoint[]): void {
    const ctx = canvas.getContext('2d')!;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!data || data.length === 0) {
      this.showEmpty(ctx, canvas.width, canvas.height);
      return;
    }

    const { top, left } = this.padding;
    const chartWidth = canvas.width - left - this.padding.right;
    const chartHeight = canvas.height - top - this.padding.bottom;

    const numLabels = data.map((d) => d.num_labels);
    const numFlips = data.map((d) => d.num_flips);
    const maxLabels = Math.max(...numLabels);
    const maxFlips = Math.max(...numFlips, 1);

    const xScale = (val: number) => left + (val / maxLabels) * chartWidth;
    const yScale = (val: number) => top + chartHeight - (val / maxFlips) * chartHeight;

    this.drawAxes(ctx, canvas.width, canvas.height);
    this.drawGrid(ctx, canvas.width, canvas.height);

    const xs = numLabels.map(xScale);
    const ys = numFlips.map(yScale);
    this.drawLine(ctx, xs, ys, this.themeColor('--color-good'));

    ctx.fillStyle = this.themeColor('--text-secondary');
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Number of Labels', canvas.width / 2, canvas.height - 10);

    ctx.save();
    ctx.translate(15, canvas.height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Prediction Flips', 0, 0);
    ctx.restore();

    ctx.textAlign = 'center';
    ctx.fillText('0', left, canvas.height - this.padding.bottom + 15);
    ctx.fillText(maxLabels.toString(), left + chartWidth, canvas.height - this.padding.bottom + 15);
    ctx.textAlign = 'right';
    ctx.fillText(maxFlips.toString(), left - 5, top + 5);
    ctx.fillText('0', left - 5, top + chartHeight + 5);
  }

  renderDiversityChart(canvas: HTMLCanvasElement, data: DiversityDataPoint[]): void {
    const ctx = canvas.getContext('2d')!;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!data || data.length === 0) {
      this.showEmpty(ctx, canvas.width, canvas.height);
      return;
    }

    const { top, left } = this.padding;
    const chartWidth = canvas.width - left - this.padding.right;
    const chartHeight = canvas.height - top - this.padding.bottom;

    const numLabels = data.map((d) => d.num_labels);
    const levels = data.map((d) => d.diversity_level);
    const treeDepth = data[0].depth;

    const maxLabels = Math.max(...numLabels);
    const maxLevel = Math.max(treeDepth, Math.max(...levels), 1);
    const minLevel = Math.min(0, Math.min(...levels));

    const xScale = (val: number) => left + (val / maxLabels) * chartWidth;
    const yScale = (val: number) =>
      top + chartHeight - ((val - minLevel) / (maxLevel - minLevel || 1)) * chartHeight;

    this.drawAxes(ctx, canvas.width, canvas.height);
    this.drawGrid(ctx, canvas.width, canvas.height);

    // Draw tree depth target line
    ctx.strokeStyle = this.themeColor('--color-good');
    ctx.lineWidth = 1;
    ctx.setLineDash([6, 4]);
    const depthY = yScale(treeDepth);
    ctx.beginPath();
    ctx.moveTo(left, depthY);
    ctx.lineTo(left + chartWidth, depthY);
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.fillStyle = this.themeColor('--color-good');
    ctx.font = '11px sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText(`depth ${treeDepth}`, left + chartWidth - 55, depthY - 5);

    const xs = numLabels.map(xScale);
    const ys = levels.map(yScale);
    this.drawLine(ctx, xs, ys, this.themeColor('--accent'));

    ctx.fillStyle = this.themeColor('--text-secondary');
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Number of Labels', canvas.width / 2, canvas.height - 10);

    ctx.save();
    ctx.translate(15, canvas.height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Diversity Level', 0, 0);
    ctx.restore();

    ctx.textAlign = 'center';
    ctx.fillText('0', left, canvas.height - this.padding.bottom + 15);
    ctx.fillText(maxLabels.toString(), left + chartWidth, canvas.height - this.padding.bottom + 15);
    ctx.textAlign = 'right';
    ctx.fillText(maxLevel.toFixed(1), left - 5, top + 5);
    ctx.fillText(minLevel.toFixed(1), left - 5, top + chartHeight + 5);
  }
}
