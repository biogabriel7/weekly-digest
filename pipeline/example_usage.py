#!/usr/bin/env python3
"""
Example usage of the Educational Analysis Pipeline
=================================================
This script demonstrates how to use the pipeline programmatically.
"""

import sys
import os
from pathlib import Path

# Add the pipeline directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from educational_analysis_pipeline import EducationalAnalysisPipeline
from modules.data_loader import DataLoader


def list_available_teachers(data_path='../top_teachers_entries.csv'):
    """List all available teacher-class combinations"""
    print("Loading data to find available teacher-class combinations...")
    
    loader = DataLoader()
    try:
        df = loader.load_data(data_path)
        teacher_classes = loader.get_available_teacher_classes(df)
        
        print("\nAvailable Teacher-Class Combinations:")
        print("-" * 60)
        print(f"{'Teacher':<25} {'Class':<20} {'Entries':<10}")
        print("-" * 60)
        
        for tc in teacher_classes[:10]:  # Show top 10
            print(f"{tc['teacher']:<25} {tc['class']:<20} {tc['entries']:<10}")
        
        if len(teacher_classes) > 10:
            print(f"\n... and {len(teacher_classes) - 10} more combinations")
        
        return teacher_classes
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return []


def run_analysis_for_teacher(teacher_name, class_name, data_path='../top_teachers_entries.csv'):
    """Run analysis for a specific teacher-class combination"""
    print(f"\nRunning analysis for: {teacher_name} - {class_name}")
    print("=" * 60)
    
    # Create pipeline instance
    pipeline = EducationalAnalysisPipeline(
        teacher_name=teacher_name,
        class_name=class_name,
        data_path=data_path
    )
    
    # Run the pipeline
    success = pipeline.run()
    
    if success:
        print(f"\nAnalysis completed successfully!")
        print(f"Results saved to: {pipeline.output_dir}")
        print(f"Open the HTML report: {pipeline.output_dir}/summary_report.html")
    else:
        print("\nAnalysis failed. Check the logs for details.")
    
    return success


def main():
    """Main example function"""
    print("Educational Analysis Pipeline - Example Usage")
    print("=" * 60)
    
    # Option 1: List available teachers
    teacher_classes = list_available_teachers()
    
    if not teacher_classes:
        print("No teacher-class combinations found.")
        return
    
    # Option 2: Run analysis for specific teachers
    print("\n\nRunning analysis for top 3 teacher-class combinations...")
    
    for i, tc in enumerate(teacher_classes[:3]):
        if i > 0:
            print("\n" + "-" * 60 + "\n")
        
        run_analysis_for_teacher(
            teacher_name=tc['teacher'],
            class_name=tc['class']
        )
    
    # Option 3: Interactive mode (commented out for automated example)
    # print("\n\nEnter a teacher-class combination to analyze:")
    # teacher = input("Teacher name: ")
    # class_name = input("Class name: ")
    # run_analysis_for_teacher(teacher, class_name)


if __name__ == "__main__":
    main()