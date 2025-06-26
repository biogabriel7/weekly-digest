# Educational Analysis Pipeline

A reusable, parameterized pipeline for analyzing teacher-class combinations and generating comprehensive visualizations and statistical insights about student performance, engagement metrics, and learning outcomes.

## Features

- **Parameterized Analysis**: Analyze any teacher-class combination by providing teacher and class names
- **Comprehensive Visualizations**: Generates multiple types of plots including trends, distributions, heatmaps, and dashboards
- **Statistical Analysis**: Performs in-depth statistical analysis including clustering and pattern detection
- **HTML Reports**: Generates beautiful HTML summary reports with embedded visualizations
- **Modular Architecture**: Clean, modular code structure for easy maintenance and extension
- **Error Handling**: Robust error handling and data validation
- **Logging**: Detailed logging for debugging and monitoring

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Basic usage:
```bash
python educational_analysis_pipeline.py --teacher "Teacher Name" --crew "Crew Name"
```

Example:
```bash
python educational_analysis_pipeline.py --teacher "Annamaria David" --crew "Lower Primary"
```

With custom data file:
```bash
python educational_analysis_pipeline.py --teacher "Owen Thomas" --crew "Year 6" --data-path "custom_data.csv"
```

With verbose logging:
```bash
python educational_analysis_pipeline.py --teacher "Elena Mari Colomar" --crew "3 - 4" --verbose
```

## Output Structure

The pipeline creates a timestamped results directory with the following structure:

```
results/
└── TeacherName_ClassName_YYYY-MM-DD_HH-MM-SS/
    ├── plots/
    │   ├── weekly_entry_trends.png
    │   ├── student_coverage_timeline.png
    │   ├── student_coverage_bars.png
    │   ├── entry_distribution.png
    │   ├── coverage_heatmap.png
    │   └── comprehensive_dashboard.png
    ├── data/
    │   ├── filtered_data.csv
    │   └── analysis_results.json
    └── summary_report.html
```

## Data Requirements

The input CSV file must contain the following columns:
- `date`: Date of the entry
- `teacher_name`: Name of the teacher
- `crew`: Class/crew name
- `student_name`: Name of the student

## Pipeline Architecture

### Modules

1. **data_loader.py**: Handles data loading, validation, and preprocessing
2. **analysis.py**: Contains all analysis modules:
   - WeeklyAnalysis: Analyzes weekly entry patterns
   - CoverageAnalysis: Analyzes student coverage patterns
   - EntryDistributionAnalysis: Analyzes entry distribution across students
   - TemporalAnalysis: Analyzes temporal patterns and trends
   - StatisticalAnalysis: Performs advanced statistical analysis including clustering

3. **visualization.py**: Contains all visualization modules:
   - WeeklyVisualization: Creates weekly trend visualizations
   - CoverageVisualization: Creates coverage visualizations
   - DistributionVisualization: Creates distribution plots
   - HeatmapVisualization: Creates coverage heatmaps
   - ComprehensiveVisualization: Creates comprehensive dashboards

4. **report_generator.py**: Generates HTML summary reports

## Analysis Components

### 1. Weekly Analysis
- Tracks entry counts per week
- Calculates statistics (mean, median, std, min, max)
- Identifies trends over time

### 2. Coverage Analysis
- Measures percentage of students with entries each week
- Classifies students into coverage patterns:
  - Consistently Covered
  - Moderately Covered
  - Sporadic Coverage
  - Missing from Records

### 3. Entry Distribution
- Analyzes distribution of entries across students
- Identifies top and bottom performers
- Calculates quartiles and statistical measures

### 4. Temporal Analysis
- Examines consistency of patterns over time
- Calculates coefficient of variation for temporal stability
- Identifies seasonal or periodic patterns

### 5. Statistical Analysis
- Performs K-means clustering on student patterns
- Calculates class-level statistics
- Identifies natural groupings in student behavior

## Visualizations

1. **Weekly Entry Trends**: Line plot showing entry counts over time
2. **Student Coverage Timeline**: Line plot showing coverage percentage over time
3. **Coverage Pattern Distribution**: Bar chart showing student distribution by coverage pattern
4. **Entry Distribution**: Box plot and histogram of entry counts per student
5. **Coverage Heatmap**: Heatmap showing weekly coverage for each student
6. **Comprehensive Dashboard**: Multi-panel dashboard with all key metrics

## HTML Report

The generated HTML report includes:
- Summary statistics
- Key findings
- Recommendations based on analysis
- All visualizations embedded as images
- Responsive design for viewing on any device

## Error Handling

The pipeline includes comprehensive error handling for:
- Missing or invalid data files
- Invalid teacher-class combinations
- Missing required columns
- Data quality issues

## Logging

The pipeline uses Python's logging module with configurable levels:
- INFO: General progress information
- DEBUG: Detailed debugging information (use --verbose flag)
- ERROR: Error messages with stack traces

## Extending the Pipeline

To add new analysis or visualization components:

1. Add new analysis class to `modules/analysis.py`
2. Add new visualization class to `modules/visualization.py`
3. Update the main pipeline to use the new components
4. Update the report generator to include new results

## License

Internal Usage.

## Support

For issues or questions, please refer to Gabriel.
