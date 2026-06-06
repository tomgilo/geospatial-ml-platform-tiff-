"""PDF report generation using matplotlib and basic PDF composition."""

import os
import io
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def generate_pdf_report(summary_df, all_metrics, config_info, training_info,
                        chart_paths, output_path, title="Geospatial ML Evaluation Report"):
    """Generate a comprehensive PDF evaluation report."""

    doc = SimpleDocTemplate(output_path, pagesize=A4,
                           rightMargin=2*cm, leftMargin=2*cm,
                           topMargin=2*cm, bottomMargin=2*cm)

    styles = getSampleStyleSheet()
    story = []

    # Try to register Chinese font
    try:
        font_paths = [
            "C:/Windows/Fonts/simhei.ttf",
            "C:/Windows/Fonts/msyh.ttc",
            "C:/Windows/Fonts/simkai.ttf",
        ]
        for fp in font_paths:
            if os.path.exists(fp):
                pdfmetrics.registerFont(TTFont('ChineseFont', fp))
                chinese_style = ParagraphStyle(
                    'ChineseStyle', parent=styles['Normal'],
                    fontName='ChineseFont', fontSize=10, leading=14
                )
                title_style = ParagraphStyle(
                    'ChineseTitle', parent=styles['Title'],
                    fontName='ChineseFont', fontSize=18, leading=22
                )
                heading_style = ParagraphStyle(
                    'ChineseHeading', parent=styles['Heading2'],
                    fontName='ChineseFont', fontSize=14, leading=18
                )
                break
        else:
            chinese_style = styles['Normal']
            title_style = styles['Title']
            heading_style = styles['Heading2']
    except Exception:
        chinese_style = styles['Normal']
        title_style = styles['Title']
        heading_style = styles['Heading2']

    # Title
    story.append(Paragraph(title, title_style))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", chinese_style))
    story.append(Spacer(1, 0.5*cm))

    # Summary metrics
    story.append(Paragraph("1. Performance Summary", heading_style))
    if summary_df is not None and not summary_df.empty:
        # Convert DataFrame to table
        table_data = [summary_df.columns.tolist()] + summary_df.values.tolist()
        # Format floats
        for i in range(1, len(table_data)):
            for j in range(len(table_data[i])):
                if isinstance(table_data[i][j], float):
                    table_data[i][j] = f"{table_data[i][j]:.4f}"

        t = Table(table_data)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8f9fa')),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
        ]))
        story.append(t)
    story.append(Spacer(1, 0.5*cm))

    # Configuration
    story.append(Paragraph("2. Configuration", heading_style))
    config_data = [["Parameter", "Value"]]
    for k, v in config_info.items():
        config_data.append([str(k), str(v)])
    t = Table(config_data, colWidths=[6*cm, 10*cm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.5*cm))

    # Charts
    if chart_paths:
        story.append(PageBreak())
        story.append(Paragraph("3. Visualizations", heading_style))
        for cp in chart_paths:
            if os.path.exists(cp) and cp.endswith('.png'):
                img = Image(cp, width=16*cm, height=10*cm)
                story.append(img)
                story.append(Spacer(1, 0.3*cm))

    doc.build(story)
    return output_path
