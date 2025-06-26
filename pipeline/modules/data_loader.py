"""
Data Loading and Preprocessing Module
=====================================
Handles data loading, validation, and preprocessing for the educational analysis pipeline.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """Handles data loading and preprocessing operations"""
    
    def __init__(self):
        """Initialize the data loader"""
        self.required_columns = ['date', 'teacher_name', 'crew', 'student_name']
    
    def load_data(self, file_path):
        """
        Load data from CSV file
        
        Args:
            file_path (str): Path to the CSV file
            
        Returns:
            pd.DataFrame: Loaded and preprocessed data
        """
        try:
            logger.info(f"Loading data from {file_path}")
            df = pd.read_csv(file_path)
            
            # Validate required columns
            missing_columns = set(self.required_columns) - set(df.columns)
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Preprocess data
            df = self._preprocess_data(df)
            
            logger.info(f"Successfully loaded {len(df)} records")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def _preprocess_data(self, df):
        """
        Preprocess the loaded data
        
        Args:
            df (pd.DataFrame): Raw data
            
        Returns:
            pd.DataFrame: Preprocessed data
        """
        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Create teacher-class identifier
        df['teacher_class'] = df['teacher_name'] + ' - ' + df['crew']
        
        # Add week information
        df['week'] = df['date'].dt.to_period('W')
        df['week_start'] = df['week'].apply(lambda x: x.start_time)
        
        # Add month information
        df['month'] = df['date'].dt.to_period('M')
        
        # Sort by date
        df = df.sort_values('date')
        
        return df
    
    def validate_teacher_class(self, df, teacher_name, class_name):
        """
        Validate if the teacher-class combination exists in the data
        
        Args:
            df (pd.DataFrame): Data to validate
            teacher_name (str): Teacher name
            class_name (str): Class name
            
        Returns:
            bool: True if combination exists, False otherwise
        """
        teacher_class = f"{teacher_name} - {class_name}"
        
        if teacher_class not in df['teacher_class'].values:
            # Check if teacher exists
            if teacher_name not in df['teacher_name'].values:
                logger.error(f"Teacher '{teacher_name}' not found in data")
                logger.info(f"Available teachers: {sorted(df['teacher_name'].unique())}")
            else:
                # Teacher exists, check their classes
                teacher_classes = df[df['teacher_name'] == teacher_name]['crew'].unique()
                logger.error(f"Class '{class_name}' not found for teacher '{teacher_name}'")
                logger.info(f"Available classes for {teacher_name}: {sorted(teacher_classes)}")
            
            return False
        
        return True
    
    def filter_by_teacher_class(self, df, teacher_name, class_name):
        """
        Filter data for specific teacher-class combination
        
        Args:
            df (pd.DataFrame): Full dataset
            teacher_name (str): Teacher name
            class_name (str): Class name
            
        Returns:
            pd.DataFrame: Filtered data
        """
        teacher_class = f"{teacher_name} - {class_name}"
        filtered_df = df[df['teacher_class'] == teacher_class].copy()
        
        logger.info(f"Filtered data for {teacher_class}: {len(filtered_df)} records")
        logger.info(f"Date range: {filtered_df['date'].min().date()} to {filtered_df['date'].max().date()}")
        logger.info(f"Number of students: {filtered_df['student_name'].nunique()}")
        logger.info(f"Number of weeks: {filtered_df['week'].nunique()}")
        
        return filtered_df
    
    def get_available_teacher_classes(self, df):
        """
        Get list of available teacher-class combinations
        
        Args:
            df (pd.DataFrame): Dataset
            
        Returns:
            list: List of teacher-class combinations
        """
        teacher_classes = df.groupby(['teacher_name', 'crew']).size().reset_index(name='entries')
        teacher_classes = teacher_classes.sort_values('entries', ascending=False)
        
        result = []
        for _, row in teacher_classes.iterrows():
            result.append({
                'teacher': row['teacher_name'],
                'class': row['crew'],
                'teacher_class': f"{row['teacher_name']} - {row['crew']}",
                'entries': row['entries']
            })
        
        return result