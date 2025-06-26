"""
Report Generator Module
======================
Generates HTML summary reports for the educational analysis pipeline.
"""

import os
from pathlib import Path
from datetime import datetime
import json
import base64
import logging

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates HTML summary reports"""
    
    def __init__(self):
        self.template = self._get_html_template()
    
    def generate_html_report(self, teacher_name, class_name, analysis_results, 
                           plots_dir, output_path):
        """
        Generate HTML summary report
        
        Args:
            teacher_name (str): Teacher name
            class_name (str): Class name
            analysis_results (dict): Analysis results
            plots_dir (Path): Directory containing plot images
            output_path (Path): Output path for HTML report
        """
        logger.info("Generating HTML report...")
        
        # Prepare data for template
        template_data = {
            'teacher_name': teacher_name,
            'class_name': class_name,
            'teacher_class': f"{teacher_name} - {class_name}",
            'generation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'summary_stats': self._prepare_summary_stats(analysis_results),
            'key_findings': self._prepare_key_findings(analysis_results),
            'recommendations': self._prepare_recommendations(analysis_results),
            'plots': self._prepare_plots(plots_dir)
        }
        
        # Generate HTML
        html_content = self.template
        for key, value in template_data.items():
            placeholder = f"{{{{{key}}}}}"
            if isinstance(value, list):
                value = '\n'.join(value)
            elif isinstance(value, dict):
                value = self._dict_to_html_table(value)
            html_content = html_content.replace(placeholder, str(value))
        
        # Write report
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"HTML report generated: {output_path}")
    
    def _get_html_template(self):
        """Get HTML template"""
        return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Educational Analysis Report - {{teacher_class}}</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .header {
            background-color: #2c3e50;
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }
        h1 {
            margin: 0;
            font-size: 2.5em;
        }
        h2 {
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
            margin-top: 30px;
        }
        h3 {
            color: #34495e;
            margin-top: 20px;
        }
        .metadata {
            font-size: 0.9em;
            opacity: 0.8;
        }
        .section {
            background: white;
            padding: 25px;
            margin-bottom: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .stat-card {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            border: 1px solid #e9ecef;
        }
        .stat-value {
            font-size: 2em;
            font-weight: bold;
            color: #3498db;
            margin: 10px 0;
        }
        .stat-label {
            color: #7f8c8d;
            font-size: 0.9em;
        }
        .findings-list {
            list-style-type: none;
            padding: 0;
        }
        .findings-list li {
            padding: 15px;
            margin: 10px 0;
            background: #ecf0f1;
            border-left: 4px solid #3498db;
            border-radius: 4px;
        }
        .recommendations {
            background: #e8f5e9;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #4caf50;
        }
        .recommendations ul {
            margin: 10px 0;
            padding-left: 20px;
        }
        .plot-gallery {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .plot-container {
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }
        .plot-container img {
            max-width: 100%;
            height: auto;
            border-radius: 4px;
        }
        .plot-title {
            font-weight: bold;
            margin: 10px 0;
            color: #2c3e50;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #3498db;
            color: white;
            font-weight: bold;
        }
        tr:hover {
            background-color: #f5f5f5;
        }
        .alert {
            padding: 15px;
            margin: 20px 0;
            border-radius: 4px;
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            color: #856404;
        }
        .footer {
            text-align: center;
            padding: 20px;
            color: #7f8c8d;
            font-size: 0.9em;
            margin-top: 40px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Educational Analysis Report</h1>
        <div class="metadata">
            <p><strong>Teacher:</strong> {{teacher_name}}</p>
            <p><strong>Class:</strong> {{class_name}}</p>
            <p><strong>Generated:</strong> {{generation_date}}</p>
        </div>
    </div>

    <div class="section">
        <h2>📊 Summary Statistics</h2>
        <div class="stats-grid">
            {{summary_stats}}
        </div>
    </div>

    <div class="section">
        <h2>🔍 Key Findings</h2>
        <ul class="findings-list">
            {{key_findings}}
        </ul>
    </div>

    <div class="section">
        <h2>💡 Recommendations</h2>
        <div class="recommendations">
            {{recommendations}}
        </div>
    </div>

    <div class="section">
        <h2>📈 Visualizations</h2>
        <div class="plot-gallery">
            {{plots}}
        </div>
    </div>

    <div class="footer">
        <p>Generated by Educational Analysis Pipeline | © 2024</p>
    </div>
</body>
</html>'''
    
    def _prepare_summary_stats(self, analysis_results):
        """Prepare summary statistics for display"""
        stats_html = []
        
        # Extract key statistics
        class_stats = analysis_results['statistical']['class_statistics']
        coverage_stats = analysis_results['coverage']['statistics']
        entry_stats = analysis_results['entry_distribution']['statistics']
        
        stats = [
            ('Total Students', class_stats['unique_students']),
            ('Total Entries', class_stats['total_entries']),
            ('Weeks Analyzed', class_stats['weeks_covered']),
            ('Avg Coverage', f"{coverage_stats['mean_coverage']:.1f}%"),
            ('Avg Entries/Student', f"{entry_stats['mean']:.1f}"),
            ('Entry Range', f"{entry_stats['min']}-{entry_stats['max']}")
        ]
        
        for label, value in stats:
            stats_html.append(f'''
            <div class="stat-card">
                <div class="stat-label">{label}</div>
                <div class="stat-value">{value}</div>
            </div>
            ''')
        
        return '\n'.join(stats_html)
    
    def _prepare_key_findings(self, analysis_results):
        """Prepare key findings for display"""
        findings = []
        
        # Coverage patterns
        patterns = analysis_results['coverage']['student_patterns']
        total_students = patterns['consistently_covered'] + patterns['moderately_covered'] + \
                        patterns['sporadic_coverage'] + patterns['missing_from_records']
        
        good_coverage = patterns['consistently_covered'] + patterns['moderately_covered']
        poor_coverage = patterns['sporadic_coverage'] + patterns['missing_from_records']
        
        findings.append(f"<li><strong>Coverage Quality:</strong> {good_coverage}/{total_students} students "
                       f"({good_coverage/total_students*100:.1f}%) have good coverage</li>")
        
        # Temporal consistency
        temporal_stats = analysis_results['temporal']['temporal_statistics']
        consistency_score = temporal_stats['consistency_score']
        
        if consistency_score > 0.7:
            consistency_level = "High"
        elif consistency_score > 0.5:
            consistency_level = "Moderate"
        else:
            consistency_level = "Low"
        
        findings.append(f"<li><strong>Temporal Consistency:</strong> {consistency_level} "
                       f"(score: {consistency_score:.2f})</li>")
        
        # Entry distribution
        entry_stats = analysis_results['entry_distribution']['statistics']
        findings.append(f"<li><strong>Entry Distribution:</strong> Mean {entry_stats['mean']:.1f} entries/student, "
                       f"with high variability (std: {entry_stats['std']:.1f})</li>")
        
        # Weekly patterns
        weekly_stats = analysis_results['weekly']['statistics']
        findings.append(f"<li><strong>Weekly Activity:</strong> Average {weekly_stats['mean']:.1f} entries/week "
                       f"over {weekly_stats['total_weeks']} weeks</li>")
        
        return '\n'.join(findings)
    
    def _prepare_recommendations(self, analysis_results):
        """Prepare recommendations based on analysis"""
        recommendations = []
        
        # Coverage-based recommendations
        patterns = analysis_results['coverage']['student_patterns']
        missing_rate = patterns['missing_from_records'] / (patterns['consistently_covered'] + 
                                                          patterns['moderately_covered'] + 
                                                          patterns['sporadic_coverage'] + 
                                                          patterns['missing_from_records'])
        
        if missing_rate > 0.3:
            recommendations.append("<strong>Priority:</strong> Many students missing from records (>30%). "
                                 "Review recording practices and identify barriers.")
        elif missing_rate > 0.1:
            recommendations.append("Some students missing from records (>10%). "
                                 "Identify and address coverage gaps.")
        
        # Coverage consistency recommendations
        coverage_stats = analysis_results['coverage']['statistics']
        if coverage_stats['mean_coverage'] < 50:
            recommendations.append("Low overall coverage rate. "
                                 "Increase frequency of student documentation.")
        elif coverage_stats['mean_coverage'] < 70:
            recommendations.append("Moderate coverage rate. "
                                 "Aim for more regular student documentation.")
        
        # Temporal consistency recommendations
        temporal_cv = analysis_results['temporal']['weekly_trend_cv']
        if temporal_cv > 1.5:
            recommendations.append("High weekly variability detected. "
                                 "Consider implementing more consistent recording schedule.")
        
        # Entry distribution recommendations
        entry_stats = analysis_results['entry_distribution']['statistics']
        if entry_stats['std'] / entry_stats['mean'] > 1.0:
            recommendations.append("High variation in student entries. "
                                 "Ensure equitable attention to all students.")
        
        if not recommendations:
            recommendations.append("Good performance overall! "
                                 "Maintain current recording practices.")
        
        # Format as HTML list
        html_list = '<ul>\n'
        for rec in recommendations:
            html_list += f'<li>{rec}</li>\n'
        html_list += '</ul>'
        
        return html_list
    
    def _prepare_plots(self, plots_dir):
        """Prepare plot images for display"""
        plot_html = []
        
        # Expected plots
        plot_files = [
            ('weekly_entry_trends.png', 'Weekly Entry Trends'),
            ('student_coverage_timeline.png', 'Student Coverage Timeline'),
            ('student_coverage_bars.png', 'Coverage Pattern Distribution'),
            ('entry_distribution.png', 'Entry Distribution Analysis'),
            ('coverage_heatmap.png', 'Student Coverage Heatmap'),
            ('comprehensive_dashboard.png', 'Comprehensive Dashboard')
        ]
        
        for filename, title in plot_files:
            plot_path = plots_dir / filename
            if plot_path.exists():
                # Read and encode image
                with open(plot_path, 'rb') as f:
                    image_data = base64.b64encode(f.read()).decode()
                
                plot_html.append(f'''
                <div class="plot-container">
                    <div class="plot-title">{title}</div>
                    <img src="data:image/png;base64,{image_data}" alt="{title}">
                </div>
                ''')
        
        return '\n'.join(plot_html)
    
    def _dict_to_html_table(self, data_dict):
        """Convert dictionary to HTML table"""
        html = '<table>\n'
        html += '<tr><th>Metric</th><th>Value</th></tr>\n'
        
        for key, value in data_dict.items():
            if isinstance(value, float):
                value = f"{value:.2f}"
            html += f'<tr><td>{key.replace("_", " ").title()}</td><td>{value}</td></tr>\n'
        
        html += '</table>'
        return html