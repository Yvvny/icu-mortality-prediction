# ICU Mortality Prediction using Machine Learning

## 1. Project Overview

This project aims to predict ICU patient mortality using machine learning techniques on the MIMIC-IV dataset. The objective is to identify high-risk patients based on clinical and demographic features.

The pipeline includes:
- Data preprocessing and merging
- Feature engineering
- Model training
- Model evaluation
- Handling class imbalance using SMOTE

---

## 2. Dataset

The project uses publicly available ICU data from the MIMIC-IV dataset.

Files used:
- patients.csv
- admissions.csv
- icustays.csv

---

## 3. Libraries and Dependencies

The project uses the following Python libraries:

- pandas
- numpy
- scikit-learn
- imbalanced-learn (SMOTE)
- matplotlib
- pytest

---

## 4. Project Structure

```
icu-mortality-project/
│
├── data/
│   ├── patients.csv
│   ├── admissions.csv
│   ├── icustays.csv
│
├── modules/
│   ├── data_processor.py
│   ├── model.py
│   ├── utils.py
│
├── tests/
│   ├── test_model.py
│
├── main.ipynb
├── README.md
```

## 5. How to Run the Project

1. Place dataset files inside the `data/` folder:
   - patients.csv
   - admissions.csv
   - icustays.csv

2. Install required libraries:
pip install pandas numpy scikit-learn imbalanced-learn matplotlib pytest

3. Open the notebook:
main.ipynb

4. Run all cells to execute the pipeline.

---

## 6. Models Used

The following models were implemented and compared:

- Logistic Regression (Baseline)
- Logistic Regression with SMOTE (to handle class imbalance)
- Random Forest Classifier

---

## 7. Evaluation Metrics

The models were evaluated using:

- Accuracy
- F1 Score
- AUROC (Area Under ROC Curve)

---

## 8. Key Results

- Logistic Regression achieved high accuracy but failed to detect mortality cases due to class imbalance.
- Applying SMOTE significantly improved recall and F1-score for the minority class.
- Random Forest provided a balance between accuracy and classification performance.

---

## 9. Advanced Python Features Used

- Lambda functions
- List comprehension
- Set operations
- Generator functions
- Built-in modules (random)

---

## 10. Testing

Pytest was used to validate:
- Data processing pipeline
- Model data splitting

Run tests using:
pytest


---

## 11. Team Members and Contributions

### Vrushabh Anil Nikhade
Email: vnikhade@stevens.edu  
Stevens ID: 20036031
- Designed and implemented data preprocessing pipeline
- Developed machine learning models
- Integrated full workflow and evaluation pipeline

### Shipeng Ren
Email: sren11@stevens.edu 
Stevens ID: 20034233  
- Model evaluation using F1-score and AUROC
- Data exploration and visualization  
- Code organization and documentation  

### Rui Yang
Email: ryang34@stevens.edu 
Stevens ID: 20028647  
- Assisted in result interpretation
- Testing and debugging  
- Code organization and documentation  

---

## 12. Conclusion

This project demonstrates the importance of handling class imbalance in healthcare datasets. The use of SMOTE significantly improves the model's ability to detect high-risk ICU patients, making the system more clinically useful.
