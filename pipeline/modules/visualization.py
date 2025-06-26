"""
Visualization Modules
====================
Contains all visualization modules for the educational data pipeline.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Set default style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')


class BaseVisualization:
    """Base class for all visualization modules"""
    
    def __init__(self):
        self.figure_dpi = 300
        self.figure_bbox = 'tight'
    
    def save_figure(self, fig, output_path):
        """Save figure to file"""
        fig.savefig(output_path, dpi=self.figure_dpi, bbox_inches=self.figure_bbox)
        plt.close(fig)
        logger.info(f"Saved plot: {output_path}")


class WeeklyVisualization(BaseVisualization):
    """Creates weekly trend visualizations"""
    
    def create_trends_plot(self, weekly_data, teacher_class, output_path):
        """Create weekly entry trends plot"""
        fig, ax = plt.subplots(figsize=(14, 6))
        
        weekly_entries = weekly_data['weekly_entries']
        stats = weekly_data['statistics']
        
        # Plot trend line
        ax.plot(weekly_entries['week_start'], weekly_entries['entries_count'], 
                marker='o', linewidth=2, markersize=6, color='blue', label='Weekly Entries')
        
        # Add mean line
        ax.axhline(y=stats['mean'], color='red', linestyle='--', alpha=0.7, 
                  label=f'Mean: {stats["mean"]:.1f}')
        
        # Add median line
        ax.axhline(y=stats['median'], color='green', linestyle='--', alpha=0.7,
                  label=f'Median: {stats["median"]:.1f}')
        
        # Customize plot
        ax.set_title(f'Weekly Entry Trends - {teacher_class}', fontsize=16, fontweight='bold')
        ax.set_xlabel('Week Starting', fontsize=12)
        ax.set_ylabel('Number of Entries', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Rotate x-axis labels
        plt.xticks(rotation=45)
        
        # Add statistics text box
        stats_text = f'Total Weeks: {stats["total_weeks"]}\n'
        stats_text += f'Range: {stats["min"]}-{stats["max"]}\n'
        stats_text += f'Std Dev: {stats["std"]:.1f}'
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        self.save_figure(fig, output_path)


class CoverageVisualization(BaseVisualization):
    """Creates coverage visualizations"""
    
    def create_coverage_timeline(self, coverage_data, teacher_class, output_path):
        """Create coverage timeline plot"""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        weekly_coverage = coverage_data['weekly_coverage']
        stats = coverage_data['statistics']
        
        # Plot coverage percentage
        ax.plot(weekly_coverage['week_start'], weekly_coverage['coverage_percentage'],
                marker='o', linewidth=2, markersize=6, color='green', label='Coverage %')
        
        # Add mean line
        ax.axhline(y=stats['mean_coverage'], color='red', linestyle='--', alpha=0.7,
                  label=f'Mean: {stats["mean_coverage"]:.1f}%')
        
        # Customize plot
        ax.set_title(f'Weekly Student Coverage - {teacher_class}', fontsize=16, fontweight='bold')
        ax.set_xlabel('Week Starting', fontsize=12)
        ax.set_ylabel('Coverage Percentage (%)', fontsize=12)
        ax.set_ylim(0, 110)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        
        # Add statistics text
        total_students = coverage_data['total_students']
        stats_text = f'Total Students: {total_students}\n'
        stats_text += f'Mean Coverage: {stats["mean_coverage"]:.1f}%\n'
        stats_text += f'Range: {stats["min_coverage"]:.1f}%-{stats["max_coverage"]:.1f}%'
        
        props = dict(boxstyle='round', facecolor='lightgreen', alpha=0.7)
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        self.save_figure(fig, output_path)
    
    def create_coverage_bars(self, coverage_data, teacher_class, output_path):
        """Create coverage pattern bar chart"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        patterns = coverage_data['student_patterns']
        
        # Prepare data for bar chart
        categories = ['Consistently\nCovered', 'Moderately\nCovered', 
                     'Sporadic\nCoverage', 'Missing from\nRecords']
        values = [
            patterns['consistently_covered'],
            patterns['moderately_covered'],
            patterns['sporadic_coverage'],
            patterns['missing_from_records']
        ]
        colors = ['#2E8B57', '#32CD32', '#FFD700', '#FF6347']
        
        # Create bars
        bars = ax.bar(categories, values, color=colors, alpha=0.8, edgecolor='black')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{value}\n({value/sum(values)*100:.1f}%)', 
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Customize plot
        ax.set_title(f'Student Coverage Patterns - {teacher_class}', 
                    fontsize=16, fontweight='bold')
        ax.set_ylabel('Number of Students', fontsize=12)
        ax.set_ylim(0, max(values) * 1.2)
        ax.grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        self.save_figure(fig, output_path)


class DistributionVisualization(BaseVisualization):
    """Creates distribution visualizations"""
    
    def create_distribution_plots(self, distribution_data, teacher_class, output_path):
        """Create entry distribution plots"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        student_entries = distribution_data['student_entries']
        stats = distribution_data['statistics']
        
        # Box plot
        ax1.boxplot(student_entries['total_entries'], vert=True, patch_artist=True,
                   boxprops=dict(facecolor='lightblue', color='darkblue'),
                   whiskerprops=dict(color='darkblue'),
                   capprops=dict(color='darkblue'),
                   medianprops=dict(color='red', linewidth=2),
                   flierprops=dict(marker='o', color='red', alpha=0.5))
        
        ax1.set_title('Entry Distribution Box Plot', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Total Entries per Student', fontsize=12)
        ax1.grid(True, axis='y', alpha=0.3)
        
        # Add quartile labels
        quartiles = [stats['min'], stats['q1'], stats['median'], stats['q3'], stats['max']]
        labels = ['Min', 'Q1', 'Median', 'Q3', 'Max']
        
        y_range = stats['max'] - stats['min']
        for q, label in zip(quartiles, labels):
            ax1.text(1.15, q, f'{label}: {q:.0f}', va='center', fontsize=10)
        
        # Histogram
        n_bins = min(20, max(5, int(np.sqrt(len(student_entries)))))
        ax2.hist(student_entries['total_entries'], bins=n_bins, 
                color='lightgreen', edgecolor='darkgreen', alpha=0.7)
        
        # Add mean and median lines
        ax2.axvline(stats['mean'], color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {stats["mean"]:.1f}')
        ax2.axvline(stats['median'], color='blue', linestyle='--', linewidth=2,
                   label=f'Median: {stats["median"]:.1f}')
        
        ax2.set_title('Entry Distribution Histogram', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Total Entries per Student', fontsize=12)
        ax2.set_ylabel('Number of Students', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add overall title
        fig.suptitle(f'Entry Distribution Analysis - {teacher_class}', 
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        self.save_figure(fig, output_path)


class HeatmapVisualization(BaseVisualization):
    """Creates heatmap visualizations"""
    
    def create_coverage_heatmap(self, df, teacher_class, output_path):
        """Create student coverage heatmap"""
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # Prepare heatmap data
        heatmap_data = self._prepare_heatmap_data(df)
        
        if heatmap_data.empty:
            logger.warning("No data for heatmap visualization")
            return
        
        # Create heatmap
        sns.heatmap(heatmap_data, 
                   ax=ax,
                   cmap='YlOrRd',
                   annot=False,
                   fmt='d',
                   cbar_kws={'label': 'Entries per Week'},
                   linewidths=0.1,
                   linecolor='white')
        
        # Customize plot
        ax.set_title(f'Weekly Student Coverage Heatmap - {teacher_class}', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Week Number', fontsize=12)
        ax.set_ylabel('Students', fontsize=12)
        
        # Handle axis labels
        n_weeks = len(heatmap_data.columns)
        if n_weeks <= 15:
            ax.set_xticklabels(heatmap_data.columns, rotation=45, ha='right', fontsize=8)
        else:
            # Show every nth week
            step = max(1, n_weeks // 10)
            tick_positions = range(0, n_weeks, step)
            ax.set_xticks(tick_positions)
            ax.set_xticklabels([heatmap_data.columns[i] for i in tick_positions], 
                              rotation=45, ha='right', fontsize=8)
        
        # Handle y-axis labels
        n_students = len(heatmap_data.index)
        if n_students > 20:
            # Show subset of student names
            step = max(1, n_students // 15)
            tick_positions = range(0, n_students, step)
            ax.set_yticks(tick_positions)
            student_labels = [str(heatmap_data.index[i])[:15] + '...' 
                            if len(str(heatmap_data.index[i])) > 15 
                            else str(heatmap_data.index[i]) 
                            for i in tick_positions]
            ax.set_yticklabels(student_labels, rotation=0, fontsize=7)
        
        # Add statistics box
        total_students = n_students
        total_weeks = n_weeks
        max_coverage = heatmap_data.max().max()
        
        stats_text = f'Students: {total_students}\n'
        stats_text += f'Weeks: {total_weeks}\n'
        stats_text += f'Max entries/week: {int(max_coverage)}'
        
        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        self.save_figure(fig, output_path)
    
    def _prepare_heatmap_data(self, df):
        """Prepare data for heatmap visualization"""
        # Group by student and week
        student_week_counts = df.groupby(['student_name', 'week_start']).size().reset_index(name='count')
        
        # Create pivot table
        heatmap_data = student_week_counts.pivot(index='student_name', 
                                                 columns='week_start', 
                                                 values='count')
        heatmap_data = heatmap_data.fillna(0)
        
        # Sort columns chronologically
        heatmap_data = heatmap_data.sort_index(axis=1)
        
        # Create week labels
        week_labels = [f"Week {i+1}" for i in range(len(heatmap_data.columns))]
        heatmap_data.columns = week_labels
        
        return heatmap_data


class ComprehensiveVisualization(BaseVisualization):
    """Creates comprehensive dashboard visualization"""
    
    def create_dashboard(self, all_results, teacher_class, output_path):
        """Create comprehensive analysis dashboard"""
        fig = plt.figure(figsize=(20, 16))
        
        # Create grid layout
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Weekly trends (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_weekly_trends(ax1, all_results['weekly'])
        
        # 2. Coverage timeline (top middle)
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_coverage_timeline(ax2, all_results['coverage'])
        
        # 3. Distribution box plot (top right)
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_distribution_box(ax3, all_results['entry_distribution'])
        
        # 4. Coverage patterns bar chart (middle left)
        ax4 = fig.add_subplot(gs[1, 0])
        self._plot_coverage_patterns(ax4, all_results['coverage'])
        
        # 5. Entry histogram (middle middle)
        ax5 = fig.add_subplot(gs[1, 1])
        self._plot_entry_histogram(ax5, all_results['entry_distribution'])
        
        # 6. Summary statistics (middle right)
        ax6 = fig.add_subplot(gs[1, 2])
        self._plot_summary_stats(ax6, all_results)
        
        # 7. Student ranking (bottom, spanning full width)
        ax7 = fig.add_subplot(gs[2, :])
        self._plot_student_ranking(ax7, all_results['entry_distribution'])
        
        # Add main title
        fig.suptitle(f'Educational Analysis Dashboard - {teacher_class}', 
                    fontsize=20, fontweight='bold')
        
        plt.tight_layout()
        self.save_figure(fig, output_path)
    
    def _plot_weekly_trends(self, ax, weekly_data):
        """Plot weekly trends on dashboard"""
        weekly_entries = weekly_data['weekly_entries']
        stats = weekly_data['statistics']
        
        ax.plot(range(len(weekly_entries)), weekly_entries['entries_count'], 
               'b-o', markersize=4)
        ax.axhline(y=stats['mean'], color='r', linestyle='--', alpha=0.7)
        
        ax.set_title('Weekly Entry Trends', fontsize=12, fontweight='bold')
        ax.set_xlabel('Week', fontsize=10)
        ax.set_ylabel('Entries', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    def _plot_coverage_timeline(self, ax, coverage_data):
        """Plot coverage timeline on dashboard"""
        weekly_coverage = coverage_data['weekly_coverage']
        
        ax.plot(range(len(weekly_coverage)), weekly_coverage['coverage_percentage'], 
               'g-o', markersize=4)
        ax.axhline(y=coverage_data['statistics']['mean_coverage'], 
                  color='r', linestyle='--', alpha=0.7)
        
        ax.set_title('Student Coverage %', fontsize=12, fontweight='bold')
        ax.set_xlabel('Week', fontsize=10)
        ax.set_ylabel('Coverage %', fontsize=10)
        ax.set_ylim(0, 110)
        ax.grid(True, alpha=0.3)
    
    def _plot_distribution_box(self, ax, distribution_data):
        """Plot distribution box plot on dashboard"""
        student_entries = distribution_data['student_entries']['total_entries'].values
        
        box = ax.boxplot(student_entries, vert=True, patch_artist=True)
        box['boxes'][0].set_facecolor('lightblue')
        
        ax.set_title('Entry Distribution', fontsize=12, fontweight='bold')
        ax.set_ylabel('Total Entries', fontsize=10)
        ax.grid(True, axis='y', alpha=0.3)
    
    def _plot_coverage_patterns(self, ax, coverage_data):
        """Plot coverage patterns on dashboard"""
        patterns = coverage_data['student_patterns']
        
        categories = ['Consistent', 'Moderate', 'Sporadic', 'Missing']
        values = [
            patterns['consistently_covered'],
            patterns['moderately_covered'],
            patterns['sporadic_coverage'],
            patterns['missing_from_records']
        ]
        colors = ['#2E8B57', '#32CD32', '#FFD700', '#FF6347']
        
        bars = ax.bar(categories, values, color=colors, alpha=0.8)
        
        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   str(value), ha='center', va='bottom', fontsize=9)
        
        ax.set_title('Coverage Patterns', fontsize=12, fontweight='bold')
        ax.set_ylabel('Students', fontsize=10)
        ax.tick_params(axis='x', labelsize=8)
    
    def _plot_entry_histogram(self, ax, distribution_data):
        """Plot entry histogram on dashboard"""
        student_entries = distribution_data['student_entries']['total_entries'].values
        
        ax.hist(student_entries, bins=10, color='lightgreen', 
               edgecolor='darkgreen', alpha=0.7)
        
        ax.axvline(distribution_data['statistics']['mean'], 
                  color='red', linestyle='--', linewidth=2)
        ax.axvline(distribution_data['statistics']['median'], 
                  color='blue', linestyle='--', linewidth=2)
        
        ax.set_title('Entry Histogram', fontsize=12, fontweight='bold')
        ax.set_xlabel('Total Entries', fontsize=10)
        ax.set_ylabel('Students', fontsize=10)
    
    def _plot_summary_stats(self, ax, all_results):
        """Plot summary statistics table on dashboard"""
        ax.axis('off')
        
        # Prepare summary data
        summary_data = []
        
        # Class statistics
        class_stats = all_results['statistical']['class_statistics']
        summary_data.append(['Total Students', str(class_stats['unique_students'])])
        summary_data.append(['Total Entries', str(class_stats['total_entries'])])
        summary_data.append(['Date Range (days)', str(class_stats['date_range_days'])])
        summary_data.append(['Weeks Covered', str(class_stats['weeks_covered'])])
        
        # Coverage statistics
        coverage_stats = all_results['coverage']['statistics']
        summary_data.append(['Mean Coverage %', f"{coverage_stats['mean_coverage']:.1f}%"])
        
        # Entry statistics
        entry_stats = all_results['entry_distribution']['statistics']
        summary_data.append(['Avg Entries/Student', f"{entry_stats['mean']:.1f}"])
        
        # Create table
        table = ax.table(cellText=summary_data,
                        colLabels=['Metric', 'Value'],
                        cellLoc='left',
                        loc='center',
                        colWidths=[0.6, 0.4])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        
        # Style the table
        for i in range(len(summary_data) + 1):
            table[(i, 0)].set_facecolor('#E8E8E8' if i % 2 == 0 else 'white')
            table[(i, 1)].set_facecolor('#E8E8E8' if i % 2 == 0 else 'white')
        
        # Header styling
        table[(0, 0)].set_facecolor('#4CAF50')
        table[(0, 1)].set_facecolor('#4CAF50')
        table[(0, 0)].set_text_props(weight='bold', color='white')
        table[(0, 1)].set_text_props(weight='bold', color='white')
        
        ax.set_title('Summary Statistics', fontsize=12, fontweight='bold', pad=20)
    
    def _plot_student_ranking(self, ax, distribution_data):
        """Plot student ranking on dashboard"""
        student_entries = distribution_data['student_entries'].head(15)
        
        # Create horizontal bar chart
        y_pos = np.arange(len(student_entries))
        ax.barh(y_pos, student_entries['total_entries'], alpha=0.8, 
               color=plt.cm.viridis(np.linspace(0.8, 0.3, len(student_entries))))
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(student_entries['student_name'], fontsize=9)
        ax.invert_yaxis()
        
        ax.set_xlabel('Total Entries', fontsize=10)
        ax.set_title('Top 15 Students by Entry Count', fontsize=12, fontweight='bold')
        ax.grid(True, axis='x', alpha=0.3)
        
        # Add value labels
        for i, (idx, row) in enumerate(student_entries.iterrows()):
            ax.text(row['total_entries'] + 0.5, i, str(row['total_entries']), 
                   va='center', fontsize=8)


# Export all visualization classes
__all__ = [
    'WeeklyVisualization',
    'CoverageVisualization',
    'DistributionVisualization',
    'HeatmapVisualization',
    'ComprehensiveVisualization'
]