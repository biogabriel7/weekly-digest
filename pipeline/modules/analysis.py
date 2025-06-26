"""
Analysis Modules
================
Contains all analysis modules for the educational data pipeline.
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


class WeeklyAnalysis:
    """Performs weekly entry analysis"""
    
    def analyze(self, df):
        """
        Analyze weekly entry patterns
        
        Args:
            df (pd.DataFrame): Filtered data for teacher-class
            
        Returns:
            dict: Analysis results
        """
        logger.debug("Starting weekly analysis")
        
        # Calculate entries per week
        weekly_entries = df.groupby('week_start').size().reset_index(name='entries_count')
        weekly_entries = weekly_entries.sort_values('week_start')
        
        # Calculate statistics
        results = {
            'weekly_entries': weekly_entries,
            'statistics': {
                'mean': weekly_entries['entries_count'].mean(),
                'median': weekly_entries['entries_count'].median(),
                'std': weekly_entries['entries_count'].std(),
                'min': weekly_entries['entries_count'].min(),
                'max': weekly_entries['entries_count'].max(),
                'total_weeks': len(weekly_entries)
            },
            'date_range': {
                'start': df['date'].min(),
                'end': df['date'].max(),
                'days': (df['date'].max() - df['date'].min()).days
            }
        }
        
        return results


class CoverageAnalysis:
    """Analyzes student coverage patterns"""
    
    def analyze(self, df):
        """
        Analyze student coverage over time
        
        Args:
            df (pd.DataFrame): Filtered data for teacher-class
            
        Returns:
            dict: Coverage analysis results
        """
        logger.debug("Starting coverage analysis")
        
        total_students = df['student_name'].nunique()
        
        # Calculate students with entries per week
        weekly_coverage = df.groupby('week_start')['student_name'].nunique().reset_index()
        weekly_coverage.columns = ['week_start', 'students_with_entries']
        weekly_coverage['coverage_percentage'] = (weekly_coverage['students_with_entries'] / total_students) * 100
        weekly_coverage = weekly_coverage.sort_values('week_start')
        
        # Analyze individual student patterns
        student_patterns = self._analyze_student_patterns(df)
        
        results = {
            'total_students': total_students,
            'weekly_coverage': weekly_coverage,
            'statistics': {
                'mean_coverage': weekly_coverage['coverage_percentage'].mean(),
                'median_coverage': weekly_coverage['coverage_percentage'].median(),
                'min_coverage': weekly_coverage['coverage_percentage'].min(),
                'max_coverage': weekly_coverage['coverage_percentage'].max(),
                'std_coverage': weekly_coverage['coverage_percentage'].std()
            },
            'student_patterns': student_patterns
        }
        
        return results
    
    def _analyze_student_patterns(self, df):
        """Analyze individual student coverage patterns"""
        # Group by student and week to count entries
        student_week_counts = df.groupby(['student_name', 'week_start']).size().reset_index(name='entry_count')
        
        # Create pivot table
        student_matrix = student_week_counts.pivot(index='student_name', columns='week_start', values='entry_count')
        student_matrix = student_matrix.fillna(0)
        
        # Calculate metrics for each student
        coverage_metrics = []
        
        for student in student_matrix.index:
            student_entries = student_matrix.loc[student]
            
            # Basic statistics
            total_entries = student_entries.sum()
            mean_entries = student_entries.mean()
            std_entries = student_entries.std()
            
            # Coverage measures
            cv = std_entries / mean_entries if mean_entries > 0 else float('inf')
            weeks_with_entries = (student_entries > 0).sum()
            total_weeks = len(student_entries)
            coverage_consistency = weeks_with_entries / total_weeks
            
            # Classification
            if coverage_consistency >= 0.8 and cv <= 0.5:
                coverage_type = 'Consistently Covered'
            elif coverage_consistency >= 0.6 and cv <= 1.0:
                coverage_type = 'Moderately Covered'
            elif coverage_consistency >= 0.3:
                coverage_type = 'Sporadic Coverage'
            else:
                coverage_type = 'Missing from Records'
            
            coverage_metrics.append({
                'student_name': student,
                'total_entries': int(total_entries),
                'mean_entries': mean_entries,
                'std_entries': std_entries,
                'coefficient_of_variation': cv,
                'coverage_consistency': coverage_consistency,
                'coverage_type': coverage_type
            })
        
        coverage_df = pd.DataFrame(coverage_metrics)
        
        # Summary by coverage type
        coverage_summary = {
            'consistently_covered': len(coverage_df[coverage_df['coverage_type'] == 'Consistently Covered']),
            'moderately_covered': len(coverage_df[coverage_df['coverage_type'] == 'Moderately Covered']),
            'sporadic_coverage': len(coverage_df[coverage_df['coverage_type'] == 'Sporadic Coverage']),
            'missing_from_records': len(coverage_df[coverage_df['coverage_type'] == 'Missing from Records']),
            'details': coverage_df
        }
        
        return coverage_summary


class EntryDistributionAnalysis:
    """Analyzes entry distribution across students"""
    
    def analyze(self, df):
        """
        Analyze entry distribution patterns
        
        Args:
            df (pd.DataFrame): Filtered data for teacher-class
            
        Returns:
            dict: Distribution analysis results
        """
        logger.debug("Starting entry distribution analysis")
        
        # Calculate total entries per student
        student_entries = df.groupby('student_name').size().reset_index(name='total_entries')
        student_entries = student_entries.sort_values('total_entries', ascending=False)
        
        # Calculate distribution statistics
        results = {
            'student_entries': student_entries,
            'statistics': {
                'students': len(student_entries),
                'mean': student_entries['total_entries'].mean(),
                'median': student_entries['total_entries'].median(),
                'std': student_entries['total_entries'].std(),
                'min': student_entries['total_entries'].min(),
                'max': student_entries['total_entries'].max(),
                'range': student_entries['total_entries'].max() - student_entries['total_entries'].min(),
                'q1': student_entries['total_entries'].quantile(0.25),
                'q3': student_entries['total_entries'].quantile(0.75),
                'iqr': student_entries['total_entries'].quantile(0.75) - student_entries['total_entries'].quantile(0.25)
            },
            'top_students': student_entries.head(5),
            'bottom_students': student_entries.tail(5)
        }
        
        return results


class TemporalAnalysis:
    """Analyzes temporal patterns and trends"""
    
    def analyze(self, df):
        """
        Analyze temporal consistency and trends
        
        Args:
            df (pd.DataFrame): Filtered data for teacher-class
            
        Returns:
            dict: Temporal analysis results
        """
        logger.debug("Starting temporal analysis")
        
        # Monthly consistency analysis
        monthly_consistency = df.groupby(['student_name', 'month']).size().reset_index(name='entries')
        
        # Calculate monthly CV for each student
        student_monthly_cv = monthly_consistency.groupby('student_name')['entries'].apply(
            lambda x: x.std() / x.mean() if x.mean() > 0 and len(x) > 1 else 0
        )
        
        # Weekly trends
        weekly_totals = df.groupby('week').size()
        weekly_trend_cv = weekly_totals.std() / weekly_totals.mean() if weekly_totals.mean() > 0 else 0
        
        results = {
            'monthly_data': monthly_consistency,
            'student_monthly_cv': student_monthly_cv,
            'average_monthly_cv': student_monthly_cv.mean(),
            'weekly_trend_cv': weekly_trend_cv,
            'temporal_statistics': {
                'total_months': df['month'].nunique(),
                'total_weeks': df['week'].nunique(),
                'consistency_score': 1 / (1 + weekly_trend_cv)  # Higher score = more consistent
            }
        }
        
        return results


class StatisticalAnalysis:
    """Performs advanced statistical analysis"""
    
    def analyze(self, df):
        """
        Perform statistical analysis including clustering
        
        Args:
            df (pd.DataFrame): Filtered data for teacher-class
            
        Returns:
            dict: Statistical analysis results
        """
        logger.debug("Starting statistical analysis")
        
        # Prepare student-level statistics
        student_stats = self._calculate_student_statistics(df)
        
        # Perform clustering if enough data
        clustering_results = None
        if len(student_stats) >= 4:
            clustering_results = self._perform_clustering(student_stats)
        
        # Calculate class-level statistics
        class_stats = self._calculate_class_statistics(df, student_stats)
        
        results = {
            'student_statistics': student_stats,
            'class_statistics': class_stats,
            'clustering': clustering_results
        }
        
        return results
    
    def _calculate_student_statistics(self, df):
        """Calculate statistics for each student"""
        student_stats = []
        
        for student in df['student_name'].unique():
            student_data = df[df['student_name'] == student]
            
            # Weekly pattern
            weekly_entries = student_data.groupby('week').size()
            
            # Calculate metrics
            total_entries = len(student_data)
            weeks_active = weekly_entries.count()
            total_weeks = df['week'].nunique()
            activity_rate = weeks_active / total_weeks
            
            # Variability
            mean_weekly = weekly_entries.mean()
            std_weekly = weekly_entries.std()
            cv = std_weekly / mean_weekly if mean_weekly > 0 else float('inf')
            
            student_stats.append({
                'student_name': student,
                'total_entries': total_entries,
                'weeks_active': weeks_active,
                'activity_rate': activity_rate,
                'mean_weekly_entries': mean_weekly,
                'cv': cv
            })
        
        return pd.DataFrame(student_stats)
    
    def _calculate_class_statistics(self, df, student_stats):
        """Calculate class-level statistics"""
        return {
            'total_entries': len(df),
            'unique_students': df['student_name'].nunique(),
            'date_range_days': (df['date'].max() - df['date'].min()).days,
            'weeks_covered': df['week'].nunique(),
            'average_entries_per_student': len(df) / df['student_name'].nunique(),
            'student_activity_distribution': {
                'mean': student_stats['activity_rate'].mean(),
                'median': student_stats['activity_rate'].median(),
                'std': student_stats['activity_rate'].std()
            }
        }
    
    def _perform_clustering(self, student_stats):
        """Perform K-means clustering on student patterns"""
        # Prepare features
        features_df = student_stats[['activity_rate', 'cv']].copy()
        
        # Remove invalid values
        valid_mask = (features_df['cv'] != float('inf')) & (~features_df['cv'].isna())
        features_df = features_df[valid_mask]
        
        if len(features_df) < 4:
            return None
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_df)
        
        # Determine optimal clusters (max 4)
        n_clusters = min(4, len(features_df) // 2)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(features_scaled)
        
        # Add cluster labels
        features_df['cluster'] = clusters
        
        # Calculate cluster statistics
        cluster_stats = []
        for i in range(n_clusters):
            cluster_data = features_df[features_df['cluster'] == i]
            cluster_center = scaler.inverse_transform(kmeans.cluster_centers_[i].reshape(1, -1))[0]
            
            # Get student names for this cluster
            cluster_indices = features_df[features_df['cluster'] == i].index
            valid_student_stats = student_stats[valid_mask].reset_index(drop=True)
            cluster_students = valid_student_stats.loc[cluster_indices, 'student_name'].tolist()
            
            cluster_stats.append({
                'cluster_id': i,
                'size': len(cluster_data),
                'avg_activity_rate': cluster_center[0],
                'avg_cv': cluster_center[1],
                'students': cluster_students
            })
        
        return {
            'n_clusters': n_clusters,
            'cluster_stats': cluster_stats,
            'clustered_data': features_df
        }


# Update the __init__.py file to export all analysis classes
__all__ = [
    'WeeklyAnalysis',
    'CoverageAnalysis',
    'EntryDistributionAnalysis',
    'TemporalAnalysis',
    'StatisticalAnalysis'
]