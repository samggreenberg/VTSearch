/**
 * VTSearch progress charts — renders error-cost, stability, and diversity
 * canvas charts for the labeling-progress modal.
 *
 * Exposed on window.VTCharts so the main IIFE in app.js can call them.
 */
(function () {
  "use strict";

  /**
   * Read a CSS custom-property value from :root.
   */
  function themeColor(varName) {
    return getComputedStyle(document.documentElement).getPropertyValue(varName).trim();
  }

  // ---------- Error-cost chart ----------

  function renderErrorCostChart(errorCostData) {
    const canvas = document.getElementById("error-cost-chart");
    const ctx = canvas.getContext("2d");

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!errorCostData || errorCostData.length === 0) {
      ctx.fillStyle = themeColor("--text-muted");
      ctx.font = "14px sans-serif";
      ctx.fillText("No data available", 20, canvas.height / 2);
      return;
    }

    // Extract data
    const numLabels = errorCostData.map(d => d.num_labels);
    const errorCosts = errorCostData.map(d => d.error_cost);

    // Chart dimensions
    const padding = { top: 20, right: 20, bottom: 40, left: 50 };
    const chartWidth = canvas.width - padding.left - padding.right;
    const chartHeight = canvas.height - padding.top - padding.bottom;

    // Scales
    const maxLabels = Math.max(...numLabels);
    const maxCost = Math.max(...errorCosts);
    const minCost = Math.min(...errorCosts);

    const xScale = (val) => padding.left + (val / maxLabels) * chartWidth;
    const yScale = (val) => padding.top + chartHeight - ((val - minCost) / (maxCost - minCost || 1)) * chartHeight;

    // Draw axes
    ctx.strokeStyle = themeColor("--border");
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padding.left, padding.top);
    ctx.lineTo(padding.left, padding.top + chartHeight);
    ctx.lineTo(padding.left + chartWidth, padding.top + chartHeight);
    ctx.stroke();

    // Draw grid lines
    ctx.strokeStyle = themeColor("--border-subtle");
    ctx.lineWidth = 1;
    for (let i = 1; i <= 5; i++) {
      const y = padding.top + (chartHeight * i) / 5;
      ctx.beginPath();
      ctx.moveTo(padding.left, y);
      ctx.lineTo(padding.left + chartWidth, y);
      ctx.stroke();
    }

    // Draw line
    ctx.strokeStyle = themeColor("--accent");
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < errorCostData.length; i++) {
      const x = xScale(numLabels[i]);
      const y = yScale(errorCosts[i]);
      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    }
    ctx.stroke();

    // Draw points
    ctx.fillStyle = themeColor("--accent");
    for (let i = 0; i < errorCostData.length; i++) {
      const x = xScale(numLabels[i]);
      const y = yScale(errorCosts[i]);
      ctx.beginPath();
      ctx.arc(x, y, 3, 0, 2 * Math.PI);
      ctx.fill();
    }

    // Labels
    ctx.fillStyle = themeColor("--text-secondary");
    ctx.font = "12px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("Number of Labels", canvas.width / 2, canvas.height - 10);

    ctx.save();
    ctx.translate(15, canvas.height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("Error Cost", 0, 0);
    ctx.restore();

    // Axis labels
    ctx.textAlign = "center";
    ctx.fillText("0", padding.left, canvas.height - padding.bottom + 15);
    ctx.fillText(maxLabels.toString(), padding.left + chartWidth, canvas.height - padding.bottom + 15);

    ctx.textAlign = "right";
    ctx.fillText(maxCost.toFixed(2), padding.left - 5, padding.top + 5);
    ctx.fillText(minCost.toFixed(2), padding.left - 5, padding.top + chartHeight + 5);
  }

  // ---------- Stability chart ----------

  function renderStabilityChart(stabilityData) {
    const canvas = document.getElementById("stability-chart");
    const ctx = canvas.getContext("2d");

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!stabilityData || stabilityData.length === 0) {
      ctx.fillStyle = themeColor("--text-muted");
      ctx.font = "14px sans-serif";
      ctx.fillText("No data available", 20, canvas.height / 2);
      return;
    }

    const dataToPlot = stabilityData;

    const numLabels = dataToPlot.map(d => d.num_labels);
    const numFlips = dataToPlot.map(d => d.num_flips);

    // Chart dimensions
    const padding = { top: 20, right: 20, bottom: 40, left: 50 };
    const chartWidth = canvas.width - padding.left - padding.right;
    const chartHeight = canvas.height - padding.top - padding.bottom;

    // Scales
    const maxLabels = Math.max(...numLabels);
    const maxFlips = Math.max(...numFlips, 1);

    const xScale = (val) => padding.left + (val / maxLabels) * chartWidth;
    const yScale = (val) => padding.top + chartHeight - (val / maxFlips) * chartHeight;

    // Draw axes
    ctx.strokeStyle = themeColor("--border");
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padding.left, padding.top);
    ctx.lineTo(padding.left, padding.top + chartHeight);
    ctx.lineTo(padding.left + chartWidth, padding.top + chartHeight);
    ctx.stroke();

    // Draw grid lines
    ctx.strokeStyle = themeColor("--border-subtle");
    ctx.lineWidth = 1;
    for (let i = 1; i <= 5; i++) {
      const y = padding.top + (chartHeight * i) / 5;
      ctx.beginPath();
      ctx.moveTo(padding.left, y);
      ctx.lineTo(padding.left + chartWidth, y);
      ctx.stroke();
    }

    // Draw line
    ctx.strokeStyle = themeColor("--color-good");
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < dataToPlot.length; i++) {
      const x = xScale(numLabels[i]);
      const y = yScale(numFlips[i]);
      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    }
    ctx.stroke();

    // Draw points
    ctx.fillStyle = themeColor("--color-good");
    for (let i = 0; i < dataToPlot.length; i++) {
      const x = xScale(numLabels[i]);
      const y = yScale(numFlips[i]);
      ctx.beginPath();
      ctx.arc(x, y, 3, 0, 2 * Math.PI);
      ctx.fill();
    }

    // Labels
    ctx.fillStyle = themeColor("--text-secondary");
    ctx.font = "12px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("Number of Labels", canvas.width / 2, canvas.height - 10);

    ctx.save();
    ctx.translate(15, canvas.height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("Prediction Flips", 0, 0);
    ctx.restore();

    // Axis labels
    ctx.textAlign = "center";
    ctx.fillText("0", padding.left, canvas.height - padding.bottom + 15);
    ctx.fillText(maxLabels.toString(), padding.left + chartWidth, canvas.height - padding.bottom + 15);

    ctx.textAlign = "right";
    ctx.fillText(maxFlips.toString(), padding.left - 5, padding.top + 5);
    ctx.fillText("0", padding.left - 5, padding.top + chartHeight + 5);
  }

  // ---------- Diversity chart ----------

  function renderDiversityChart(diversityData) {
    const canvas = document.getElementById("diversity-chart");
    const ctx = canvas.getContext("2d");

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!diversityData || diversityData.length === 0) {
      ctx.fillStyle = themeColor("--text-muted");
      ctx.font = "14px sans-serif";
      ctx.fillText("No data available", 20, canvas.height / 2);
      return;
    }

    const numLabels = diversityData.map(d => d.num_labels);
    const levels = diversityData.map(d => d.diversity_level);
    const treeDepth = diversityData[0].depth;

    const padding = { top: 20, right: 20, bottom: 40, left: 50 };
    const chartWidth = canvas.width - padding.left - padding.right;
    const chartHeight = canvas.height - padding.top - padding.bottom;

    const maxLabels = Math.max(...numLabels);
    const maxLevel = Math.max(treeDepth, Math.max(...levels), 1);
    const minLevel = Math.min(0, Math.min(...levels));

    const xScale = (val) => padding.left + (val / maxLabels) * chartWidth;
    const yScale = (val) => padding.top + chartHeight - ((val - minLevel) / (maxLevel - minLevel || 1)) * chartHeight;

    // Draw axes
    ctx.strokeStyle = themeColor("--border");
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padding.left, padding.top);
    ctx.lineTo(padding.left, padding.top + chartHeight);
    ctx.lineTo(padding.left + chartWidth, padding.top + chartHeight);
    ctx.stroke();

    // Draw grid lines
    ctx.strokeStyle = themeColor("--border-subtle");
    ctx.lineWidth = 1;
    for (let i = 1; i <= 5; i++) {
      const y = padding.top + (chartHeight * i) / 5;
      ctx.beginPath();
      ctx.moveTo(padding.left, y);
      ctx.lineTo(padding.left + chartWidth, y);
      ctx.stroke();
    }

    // Draw tree depth target line
    ctx.strokeStyle = themeColor("--color-good");
    ctx.lineWidth = 1;
    ctx.setLineDash([6, 4]);
    const depthY = yScale(treeDepth);
    ctx.beginPath();
    ctx.moveTo(padding.left, depthY);
    ctx.lineTo(padding.left + chartWidth, depthY);
    ctx.stroke();
    ctx.setLineDash([]);

    // Label the target line
    ctx.fillStyle = themeColor("--color-good");
    ctx.font = "11px sans-serif";
    ctx.textAlign = "left";
    ctx.fillText(`depth ${treeDepth}`, padding.left + chartWidth - 55, depthY - 5);

    // Draw line
    ctx.strokeStyle = themeColor("--accent");
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < diversityData.length; i++) {
      const x = xScale(numLabels[i]);
      const y = yScale(levels[i]);
      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    }
    ctx.stroke();

    // Draw points
    ctx.fillStyle = themeColor("--accent");
    for (let i = 0; i < diversityData.length; i++) {
      const x = xScale(numLabels[i]);
      const y = yScale(levels[i]);
      ctx.beginPath();
      ctx.arc(x, y, 3, 0, 2 * Math.PI);
      ctx.fill();
    }

    // Labels
    ctx.fillStyle = themeColor("--text-secondary");
    ctx.font = "12px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("Number of Labels", canvas.width / 2, canvas.height - 10);

    ctx.save();
    ctx.translate(15, canvas.height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("Diversity Level", 0, 0);
    ctx.restore();

    // Axis labels
    ctx.textAlign = "center";
    ctx.fillText("0", padding.left, canvas.height - padding.bottom + 15);
    ctx.fillText(maxLabels.toString(), padding.left + chartWidth, canvas.height - padding.bottom + 15);

    ctx.textAlign = "right";
    ctx.fillText(maxLevel.toFixed(1), padding.left - 5, padding.top + 5);
    ctx.fillText(minLevel.toFixed(1), padding.left - 5, padding.top + chartHeight + 5);
  }

  // ---------- Public API ----------

  window.VTCharts = {
    renderErrorCostChart: renderErrorCostChart,
    renderStabilityChart: renderStabilityChart,
    renderDiversityChart: renderDiversityChart,
  };
})();
