#!/usr/bin/env python3
"""
Educational Data Analysis Pipeline
==================================
A reusable, parameterized pipeline for analyzing teacher-class combinations.
Generates comprehensive visualizations and statistical insights about student 
performance, engagement metrics, and learning outcomes.

Usage:
    python educational_analysis_pipeline.py --teacher "Teacher Name" --crew "Crew Name"

Example:
    python educational_analysis_pipeline.py --teacher "Annamaria David" --crew "Lower Primary"
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import logging
from pathlib import Path
import json
from tqdm import tqdm

# Import our custom modules
from modules.data_loader import DataLoader
from modules.analysis import (
    WeeklyAnalysis, 
    CoverageAnalysis, 
    EntryDistributionAnalysis,
    TemporalAnalysis,
    StatisticalAnalysis
)
from modules.visualization import (
    WeeklyVisualization,
    CoverageVisualization,
    DistributionVisualization,
    HeatmapVisualization,
    ComprehensiveVisualization
)
from modules.report_generator import ReportGenerator

# Configure warnings and logging
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EducationalAnalysisPipeline:
    """Main pipeline class for educational data analysis"""
    
    def __init__(self, teacher_name, class_name, data_path='all_schools_data_with_broad_categories.csv'):
        """
        Initialize the pipeline with teacher and class parameters
        
        Args:
            teacher_name (str): Name of the teacher to analyze
            class_name (str): Name of the class to analyze
            data_path (str): Path to the input data file
        """
        self.teacher_name = teacher_name
        self.class_name = class_name
        self.data_path = data_path
        self.teacher_class = f"{teacher_name} - {class_name}"
        
        # Create output directory structure
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.output_dir = Path(f"results/{teacher_name.replace(' ', '_')}_{class_name.replace(' ', '_')}_{timestamp}")
        self.plots_dir = self.output_dir / "plots"
        self.data_dir = self.output_dir / "data"
        
        # Create directories
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized pipeline for {self.teacher_class}")
        logger.info(f"Output directory: {self.output_dir}")
        
        # Initialize components
        self.data_loader = DataLoader()
        self.data = None
        self.filtered_data = None
        self.analysis_results = {}
        
    def load_and_validate_data(self):
        """Load data and validate teacher-class combination exists"""
        logger.info("Loading and validating data...")
        
        try:
            # Load data
            self.data = self.data_loader.load_data(self.data_path)
            
            # Validate teacher-class combination
            if not self.data_loader.validate_teacher_class(self.data, self.teacher_name, self.class_name):
                raise ValueError(f"No data found for {self.teacher_class}")
            
            # Filter data for specified teacher-class
            self.filtered_data = self.data_loader.filter_by_teacher_class(
                self.data, self.teacher_name, self.class_name
            )
            
            logger.info(f"Successfully loaded {len(self.filtered_data)} entries for {self.teacher_class}")
            
            # Save filtered data
            self.filtered_data.to_csv(self.data_dir / 'filtered_data.csv', index=False)
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return False
    
    def run_analyses(self):
        """Execute all analysis modules"""
        logger.info("Running analyses...")
        
        # 1. Weekly Analysis
        logger.info("Performing weekly analysis...")
        weekly_analyzer = WeeklyAnalysis()
        self.analysis_results['weekly'] = weekly_analyzer.analyze(self.filtered_data)
        
        # 2. Coverage Analysis
        logger.info("Performing coverage analysis...")
        coverage_analyzer = CoverageAnalysis()
        self.analysis_results['coverage'] = coverage_analyzer.analyze(self.filtered_data)
        
        # 3. Entry Distribution Analysis
        logger.info("Performing entry distribution analysis...")
        entry_analyzer = EntryDistributionAnalysis()
        self.analysis_results['entry_distribution'] = entry_analyzer.analyze(self.filtered_data)
        
        # 4. Temporal Analysis
        logger.info("Performing temporal analysis...")
        temporal_analyzer = TemporalAnalysis()
        self.analysis_results['temporal'] = temporal_analyzer.analyze(self.filtered_data)
        
        # 5. Statistical Analysis
        logger.info("Performing statistical analysis...")
        stat_analyzer = StatisticalAnalysis()
        self.analysis_results['statistical'] = stat_analyzer.analyze(self.filtered_data)
        
        # Save analysis results
        with open(self.data_dir / 'analysis_results.json', 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            def convert_types(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, pd.DataFrame):
                    return obj.to_dict('records')
                elif isinstance(obj, pd.Series):
                    return obj.to_dict()
                elif isinstance(obj, (pd.Timestamp, datetime)):
                    return obj.isoformat()
                elif hasattr(obj, '__dict__'):
                    return str(obj)
                return obj
            
            # Convert all DataFrames to dicts before JSON serialization
            def prepare_for_json(obj):
                if isinstance(obj, dict):
                    return {k: prepare_for_json(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [prepare_for_json(v) for v in obj]
                elif isinstance(obj, pd.DataFrame):
                    # Convert DataFrame to dict and handle timestamps
                    df_dict = obj.to_dict('records')
                    return prepare_for_json(df_dict)
                elif isinstance(obj, pd.Series):
                    # Convert Series to dict and handle timestamps
                    series_dict = obj.to_dict()
                    return prepare_for_json(series_dict)
                elif isinstance(obj, (pd.Timestamp, datetime)):
                    return obj.isoformat()
                elif isinstance(obj, pd.Period):
                    return str(obj)
                elif isinstance(obj, (np.integer, np.int64)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float64)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif pd.isna(obj):
                    return None
                else:
                    return obj
            
            # Prepare results for JSON serialization
            serializable_results = prepare_for_json(self.analysis_results)
            
            json.dump(serializable_results, f, indent=2)
        
        logger.info("All analyses completed successfully")
    
    def generate_visualizations(self):
        """Generate all visualizations"""
        logger.info("Generating visualizations...")
        
        # 1. Weekly Trends
        logger.info("Creating weekly trend visualizations...")
        weekly_viz = WeeklyVisualization()
        weekly_viz.create_trends_plot(
            self.analysis_results['weekly'], 
            self.teacher_class,
            self.plots_dir / 'weekly_entry_trends.png'
        )
        
        # 2. Coverage Visualizations
        logger.info("Creating coverage visualizations...")
        coverage_viz = CoverageVisualization()
        coverage_viz.create_coverage_timeline(
            self.analysis_results['coverage'],
            self.teacher_class,
            self.plots_dir / 'student_coverage_timeline.png'
        )
        coverage_viz.create_coverage_bars(
            self.analysis_results['coverage'],
            self.teacher_class,
            self.plots_dir / 'student_coverage_bars.png'
        )
        
        # 3. Distribution Visualizations
        logger.info("Creating distribution visualizations...")
        dist_viz = DistributionVisualization()
        dist_viz.create_distribution_plots(
            self.analysis_results['entry_distribution'],
            self.teacher_class,
            self.plots_dir / 'entry_distribution.png'
        )
        
        # 4. Heatmap Visualization
        logger.info("Creating heatmap visualization...")
        heatmap_viz = HeatmapVisualization()
        heatmap_viz.create_coverage_heatmap(
            self.filtered_data,
            self.teacher_class,
            self.plots_dir / 'coverage_heatmap.png'
        )
        
        # 5. Comprehensive Dashboard
        logger.info("Creating comprehensive dashboard...")
        comprehensive_viz = ComprehensiveVisualization()
        comprehensive_viz.create_dashboard(
            self.analysis_results,
            self.teacher_class,
            self.plots_dir / 'comprehensive_dashboard.png'
        )
        
        logger.info("All visualizations generated successfully")
    
    def generate_report(self):
        """Generate HTML summary report"""
        logger.info("Generating summary report...")
        
        report_gen = ReportGenerator()
        report_path = self.output_dir / 'summary_report.html'
        
        report_gen.generate_html_report(
            teacher_name=self.teacher_name,
            class_name=self.class_name,
            analysis_results=self.analysis_results,
            plots_dir=self.plots_dir,
            output_path=report_path
        )
        
        logger.info(f"Summary report generated: {report_path}")
    
    def run(self):
        """Execute the complete pipeline"""
        logger.info("=" * 60)
        logger.info(f"Starting Educational Analysis Pipeline")
        logger.info(f"Teacher: {self.teacher_name}")
        logger.info(f"Class: {self.class_name}")
        logger.info("=" * 60)
        
        try:
            # Step 1: Load and validate data
            if not self.load_and_validate_data():
                logger.error("Failed to load data. Exiting pipeline.")
                return False
            
            # Step 2: Run analyses
            self.run_analyses()
            
            # Step 3: Generate visualizations
            self.generate_visualizations()
            
            # Step 4: Generate report
            self.generate_report()
            
            logger.info("=" * 60)
            logger.info("Pipeline completed successfully!")
            logger.info(f"Results saved to: {self.output_dir}")
            logger.info("=" * 60)
            
            return True
            
        except Exception as e:
            logger.error(f"Pipeline failed with error: {str(e)}")
            logger.error("Stack trace:", exc_info=True)
            return False


def main():
    """Main entry point for the pipeline"""
    parser = argparse.ArgumentParser(
        description='Educational Data Analysis Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python educational_analysis_pipeline.py --teacher "Annamaria David" --crew "Lower Primary"
  python educational_analysis_pipeline.py --teacher "Owen Thomas" --crew "Year 6" --data-path "custom_data.csv"
        """
    )
    
    parser.add_argument('--teacher', type=str, required=True,
                        help='Name of the teacher to analyze')
    parser.add_argument('--crew', type=str, required=True,
                        help='Name of the crew/class to analyze')
    parser.add_argument('--data-path', type=str, default='all_schools_data_with_broad_categories.csv',
                        help='Path to the input data file (default: all_schools_data_with_broad_categories.csv)')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create and run pipeline
    pipeline = EducationalAnalysisPipeline(
        teacher_name=args.teacher,
        class_name=args.crew,
        data_path=args.data_path
    )
    
    success = pipeline.run()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()