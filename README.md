# Python-Based ICU Mortality Prediction and Patient Data Analysis System

## Project Overview
This project focuses on predicting ICU patient mortality using Python and machine learning.  
It uses demographic and ICU stay information from the MIMIC-IV dataset to identify patterns related to survival outcomes and support data-driven clinical insights.

## Problem Statement
Predicting outcomes in Intensive Care Units (ICUs) is a critical healthcare challenge. Hospitals collect large volumes of patient data, but extracting practical insights from this data requires structured analysis and predictive modeling.

The goal of this project is to:
- Analyze ICU patient data.
- Build reliable mortality prediction models.
- Visualize key trends for interpretation and decision support.

## Dataset
- **Source:** MIMIC-IV (public ICU electronic health record dataset)
- **Input format:** CSV files
- **Core fields (planned):** age, gender, ICU unit, length of stay, mortality outcome

## Technical Plan

### Main Classes
1. **PatientDataProcessor**
   - Loads raw data
   - Cleans and merges datasets
   - Performs feature engineering
   - Planned methods: `load_data()`, `clean_data()`, `merge_data()`, `feature_engineering()`

2. **MortalityPredictor**
   - Trains machine learning models
   - Evaluates predictive performance
   - Planned methods: `train_logistic_regression()`, `train_random_forest()`, `evaluate_model()`

### Core Analytical Functions
- `visualize_mortality_by_age()`
- `calculate_icu_length_of_stay_statistics()`

### Planned Libraries
- Pandas (data processing)
- NumPy (numerical operations)
- Matplotlib / Seaborn (visualization)
- Scikit-learn (machine learning)

## Software Design Requirements
- Exception handling for:
  - Missing or malformed files
  - Invalid data or model training issues
- Pytest-based unit tests for:
  - Data loading correctness
  - Prediction functionality
- Use of control flow (loops and conditions) for processing logic
- Use of mutable and immutable Python data types
- Operator overloading through methods such as `__str__()` and `__len__()`
- Docstrings and inline comments for maintainability

## Advanced Python Features (Planned)
- Lambda functions
- List comprehensions
- Built-in modules (for example, `math`, `time`)
- Generators for large-data iteration
- Set operations for unique ICU unit/patient group analysis

## Project Timeline
1. **Week 1:** Finalize design, set up GitHub, implement data loading and preprocessing
2. **Week 2:** Build classes and core functions, perform EDA and visualization
3. **Week 3:** Implement ML models, add exception handling and operator overloading
4. **Week 4:** Write Pytest tests, refactor into modules
5. **Week 5:** Final testing, debugging, documentation, and submission

## Repository Note
Proposal and related class materials are currently stored in the `AAI 551/` directory, including `Project Proposal.pdf`.
