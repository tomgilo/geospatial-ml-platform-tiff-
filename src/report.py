"""HTML evaluation report generator."""

import os
import time
from datetime import datetime
import numpy as np
import pandas as pd

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Geospatial ML — Evaluation Report</title>
<style>
  body { font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif; margin: 40px; background: #f5f7fa; color: #333; }
  h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
  h2 { color: #34495e; margin-top: 30px; border-bottom: 1px solid #ddd; padding-bottom: 5px; }
  table { border-collapse: collapse; width: 100%; margin: 15px 0; background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
  th { background: #3498db; color: white; padding: 10px 12px; text-align: left; }
  td { padding: 8px 12px; border-bottom: 1px solid #eee; }
  tr:hover { background: #f8f9fa; }
  .best { background: #d4edda; font-weight: bold; }
  .metric-card { display: inline-block; background: white; border-radius: 8px; padding: 15px 25px; margin: 8px;
                 box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center; min-width: 120px; }
  .metric-value { font-size: 28px; font-weight: bold; color: #3498db; }
  .metric-label { font-size: 12px; color: #888; text-transform: uppercase; margin-top: 5px; }
  .warning { background: #fff3cd; border-left: 4px solid #ffc107; padding: 10px 15px; margin: 10px 0; }
  img { max-width: 100%; height: auto; margin: 10px 0; border-radius: 4px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
  .footer { margin-top: 40px; color: #999; font-size: 12px; border-top: 1px solid #ddd; padding-top: 10px; }
</style>
</head>
<body>

<h1>Geospatial TIFF Machine Learning — Evaluation Report</h1>
<p>Generated: {{generated_at}}</p>

<div class="metrics-row">
  <div class="metric-card"><div class="metric-value">{{n_pixels}}</div><div class="metric-label">Valid Pixels</div></div>
  <div class="metric-card"><div class="metric-value">{{n_models}}</div><div class="metric-label">Models Trained</div></div>
  <div class="metric-card"><div class="metric-value">{{train_years}}</div><div class="metric-label">Training Period</div></div>
  <div class="metric-card"><div class="metric-value">{{predict_years}}</div><div class="metric-label">Prediction Period</div></div>
  <div class="metric-card"><div class="metric-value">{{total_time}}</div><div class="metric-label">Total Time (s)</div></div>
</div>

<h2>1. Model Performance Summary</h2>
{{summary_table}}

<h2>2. Best Model</h2>
<p><strong>Best R²:</strong> {{best_model_r2}} (R² = {{best_r2_value}})</p>
<p><strong>Best RMSE:</strong> {{best_model_rmse}} (RMSE = {{best_rmse_value}})</p>

<h2>3. Configurations</h2>
{{config_table}}

<h2>4. Training Details</h2>
{{training_details}}

<h2>5. Prediction Outputs</h2>
{{prediction_outputs}}

<h2>6. Visualization</h2>
{{chart_images}}

{{warnings_section}}

<div class="footer">Geospatial ML Application — Report auto-generated</div>
</body>
</html>"""


def generate_html_report(summary_df, all_metrics, config_info, training_info,
                         prediction_outputs, chart_paths, warnings,
                         output_path, total_time=0):
    """Generate a complete HTML evaluation report."""

    # Build summary table
    if summary_df is not None and not summary_df.empty:
        summary_table = summary_df.to_html(
            classes="summary-table", index=False, float_format="%.4f",
            border=0
        )
        # Highlight best model row (guard against all-NA)
        if "mean_r2" in summary_df.columns and summary_df["mean_r2"].notna().any():
            best_r2_idx = summary_df["mean_r2"].idxmax()
        else:
            best_r2_idx = None
        if "mean_rmse" in summary_df.columns and summary_df["mean_rmse"].notna().any():
            best_rmse_idx = summary_df["mean_rmse"].idxmin()
        else:
            best_rmse_idx = None

        # Add a visible warning if every model failed
        if best_r2_idx is None and best_rmse_idx is None:
            summary_table = (
                '<div class="warning">&#9888; All models failed to produce valid metrics. '
                'Please check your training data, time ranges, and ensure there are enough valid pixels/samples.</div>\n'
                + summary_table
            )
    else:
        summary_table = "<p>No evaluation data available.</p>"
        best_r2_idx = None
        best_rmse_idx = None

    # Best model info
    if summary_df is not None and not summary_df.empty and best_r2_idx is not None:
        best_r2_model = summary_df.loc[best_r2_idx, "model"]
        best_r2_val = summary_df.loc[best_r2_idx, "mean_r2"]
        best_rmse_model = summary_df.loc[best_rmse_idx, "model"]
        best_rmse_val = summary_df.loc[best_rmse_idx, "mean_rmse"]
    else:
        best_r2_model = best_r2_val = best_rmse_model = best_rmse_val = "N/A"

    # Config table
    config_rows = ""
    for k, v in config_info.items():
        config_rows += f"<tr><td><strong>{k}</strong></td><td>{v}</td></tr>\n"
    config_table = f"<table><tr><th>Parameter</th><th>Value</th></tr>{config_rows}</table>"

    # Training details
    training_rows = ""
    for k, v in training_info.items():
        training_rows += f"<tr><td><strong>{k}</strong></td><td>{v}</td></tr>\n"
    training_details = f"<table><tr><th>Parameter</th><th>Value</th></tr>{training_rows}</table>"

    # Prediction outputs
    pred_html = "<ul>"
    for model_name, files in prediction_outputs.items():
        pred_html += f"<li><strong>{model_name}:</strong> {len(files)} file(s)"
        pred_html += "<ul>"
        for f in files[:10]:  # limit display
            pred_html += f"<li><code>{os.path.basename(f)}</code></li>"
        if len(files) > 10:
            pred_html += f"<li>... and {len(files) - 10} more</li>"
        pred_html += "</ul></li>"
    pred_html += "</ul>"

    # Chart images (embedded as relative paths)
    chart_html = ""
    for cp in chart_paths:
        rel_path = os.path.basename(cp)
        chart_html += f'<img src="{rel_path}" alt="{rel_path}" style="max-width:600px; margin:10px;">\n'

    # Warnings
    warnings_html = ""
    if warnings:
        warnings_html = "<h2>7. Warnings</h2><ul>"
        for w in warnings:
            warnings_html += f'<li class="warning">{w}</li>'
        warnings_html += "</ul>"

    # Fill template
    html = HTML_TEMPLATE.replace("{{generated_at}}", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    html = html.replace("{{n_pixels}}", str(training_info.get("n_valid_pixels", "N/A")))
    html = html.replace("{{n_models}}", str(len(prediction_outputs)))
    html = html.replace("{{train_years}}", str(config_info.get("train_period", "N/A")))
    html = html.replace("{{predict_years}}", str(config_info.get("predict_period", "N/A")))
    html = html.replace("{{total_time}}", f"{total_time:.1f}")
    html = html.replace("{{summary_table}}", summary_table)
    html = html.replace("{{best_model_r2}}", str(best_r2_model))
    html = html.replace("{{best_r2_value}}", f"{best_r2_val:.4f}" if isinstance(best_r2_val, float) else str(best_r2_val))
    html = html.replace("{{best_model_rmse}}", str(best_rmse_model))
    html = html.replace("{{best_rmse_value}}", f"{best_rmse_val:.4f}" if isinstance(best_rmse_val, float) else str(best_rmse_val))
    html = html.replace("{{config_table}}", config_table)
    html = html.replace("{{training_details}}", training_details)
    html = html.replace("{{prediction_outputs}}", pred_html)
    html = html.replace("{{chart_images}}", chart_html)
    html = html.replace("{{warnings_section}}", warnings_html)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    return output_path
