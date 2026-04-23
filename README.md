# ICU Mortality Prediction using Machine Learning

## 1. Project Overview

This project predicts ICU patient mortality using machine learning on MIMIC-IV style tabular data. The workflow is organized as a notebook-driven pipeline with reusable Python modules for data processing, model training, evaluation, and visualization.

The project includes:

- data loading and merging
- data cleaning and feature engineering
- model training and evaluation
- class imbalance handling with SMOTE / SMOTENC
- notebook-based execution through `main.ipynb`

## 2. Dataset

The project uses ICU-related tables derived from the MIMIC-IV dataset structure.

Files used by the current workflow:

- `patients.csv`
- `admissions.csv`
- `icustays.csv`

For the reproducible public version of this repository, the `data/` folder uses the **MIMIC-IV Clinical Database Demo v2.2** files with the same table structure as the full dataset.

Demo source:

- `patients.csv.gz` from `hosp/`
- `admissions.csv.gz` from `hosp/`
- `icustays.csv.gz` from `icu/`

Official demo dataset:

- https://physionet.org/content/mimic-iv-demo/2.2/

If you have authorized access to the full MIMIC-IV dataset, you can replace the demo CSV files with the full v2.2 versions using the same filenames and schema. The full dataset is not distributed in this repository.

## 3. Libraries and Dependencies

Core libraries used in this project:

- `pandas`
- `numpy`
- `scikit-learn`
- `imbalanced-learn`
- `matplotlib`
- `pytest`
- `notebook`

Install all dependencies with:

```bash
python -m pip install -r requirements.txt
```

## 4. Project Structure

```text
icu-mortality-prediction/
|-- data/
|   |-- patients.csv
|   |-- admissions.csv
|   `-- icustays.csv
|-- modules/
|   |-- data_processor.py
|   |-- model.py
|   `-- utils.py
|-- tests/
|   |-- test_model.py
|   `-- test_model_synthetic.py
|-- main.ipynb
|-- README.md
`-- requirements.txt
```

## 5. How to Run the Project

1. The repository already includes the demo v2.2 CSV files inside the `data/` folder for reproducible execution.
2. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

3. Run the notebook:

```bash
python -m notebook main.ipynb
```

4. Run all notebook cells to execute the full workflow.

To use the full restricted-access MIMIC-IV dataset instead of the demo data, replace the three CSV files in `data/` with the full v2.2 versions while keeping the same filenames.

## 6. Classes and Program Design

The project uses two meaningful classes:

### `PatientDataProcessor`

Responsible for:

- loading CSV files
- merging patient, admission, and ICU stay data
- cleaning required fields
- engineering modeling features

### `MortalityPredictor`

Responsible for:

- receiving processed data from `PatientDataProcessor`
- splitting train and test data
- training machine learning models
- evaluating model performance

The class relationship is **composition**: `MortalityPredictor` can be initialized with a `PatientDataProcessor` object and uses the processed modeling data produced by that class.

## 7. Functions

The project includes multiple meaningful functions, including:

- `plot_mortality_distribution(df)`
- `plot_model_f1_scores(score_dict)`

These functions support data interpretation and model comparison.

## 8. Models Used

The following models are implemented:

- Logistic Regression
- Logistic Regression with SMOTE / SMOTENC
- Random Forest Classifier

## 9. Evaluation Metrics

The models are evaluated using:

- Accuracy
- F1 Score
- AUROC
- Confusion Matrix
- Classification Report

## 10. Advanced Python Features Used

To satisfy the project requirements, the code uses:

- lambda functions
- list comprehension
- set operations
- generator functions
- built-in module `random`
- operator overloading with `__str__()` and `__len__()`

## 11. Exception Handling

The code includes multiple exception-handling scenarios, including:

- missing data files during loading
- missing required columns in merged data
- model training before calling `split_data()`
- missing `mortality` target column in model input

## 12. Testing

Pytest is used to validate:

- the processed modeling columns
- train/test data splitting
- exception handling behavior
- synthetic-data execution without relying on external CSV files

Run tests with:

```bash
python -m pytest -q
```

## 13. Team Members and Contributions

### Vrushabh Anil Nikhade
Email: vnikhade@stevens.edu  
Stevens ID: 20036031

- designed and implemented the data preprocessing pipeline
- developed the machine learning models
- integrated the workflow into the project notebook

### Shipeng Ren
Email: sren11@stevens.edu  
Stevens ID: 20034233

- supported model evaluation using F1-score and AUROC
- assisted with data exploration and visualization
- contributed to code organization and documentation
- Updated by joking000

### Rui Yang
Email: ryang34@stevens.edu  
Stevens ID: 20028647

- assisted with result interpretation
- contributed to testing and debugging
- supported code organization and documentation

## 14. Conclusion

This project demonstrates a complete notebook-based machine learning workflow for ICU mortality prediction using modular Python code. It highlights data preprocessing, feature engineering, imbalance handling, evaluation, and testing in a structure aligned with the course programming requirements.
