import pandas as pd
import pytest

from modules.data_processor import PatientDataProcessor
from modules.model import MortalityPredictor


def build_synthetic_merged_df():
    """
    Build a small merged dataframe for testing the processor without CSV files.
    """
    return pd.DataFrame(
        {
            "anchor_age": [52, 73, 55, 46, None],
            "anchor_year": [2180, 2186, 2157, 2165, 2172],
            "gender": ["F", "M", "F", "M", "F"],
            "los": [0.4, 0.5, 1.1, 1.3, 2.0],
            "hospital_expire_flag": [0, 1, 0, 1, 0],
            "first_careunit": ["MICU", "SICU", "MICU", "CCU", "MICU"],
            "admittime": [
                "2180-07-23 12:35:00",
                "2189-06-27 07:38:00",
                "2157-11-18 22:56:00",
                "2165-01-12 08:40:00",
                "2172-04-03 10:15:00",
            ],
            "dischtime": [
                "2180-07-25 17:55:00",
                "2189-07-03 03:00:00",
                "2157-11-25 18:00:00",
                "2165-01-18 14:10:00",
                "2172-04-06 09:45:00",
            ],
            "deathtime": [None, "2189-06-30 01:00:00", None, None, None],
            "intime": [
                "2180-07-23 14:00:00",
                "2189-06-27 08:42:00",
                "2157-11-20 19:18:02",
                "2165-01-13 10:05:00",
                "2172-04-03 12:00:00",
            ],
            "outtime": [
                "2180-07-23 23:50:47",
                "2189-06-27 20:38:27",
                "2157-11-21 22:08:00",
                "2165-01-15 17:25:00",
                "2172-04-05 07:10:00",
            ],
            "dod": ["2180-09-09", None, None, None, None],
        }
    )


def build_synthetic_model_df():
    """
    Build a balanced modeling dataframe for testing the predictor.
    """
    return pd.DataFrame(
        {
            "admission_age": [45, 51, 62, 70, 38, 59, 66, 74, 41, 57],
            "gender_encoded": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            "los": [1.2, 2.1, 0.9, 3.0, 1.5, 2.6, 1.1, 3.4, 0.8, 2.2],
            "mortality": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        }
    )


def test_data_processor_synthetic_model_columns():
    """
    Test that the processor produces the expected modeling columns from synthetic data.
    """
    processor = PatientDataProcessor()
    processor.merged_df = build_synthetic_merged_df()

    processor.clean_data()
    processor.feature_engineering()

    df = processor.get_model_data()

    assert list(df.columns) == ["admission_age", "gender_encoded", "los", "mortality"]
    assert len(df) == 4
    assert set(df["mortality"].unique()) == {0, 1}
    assert df.iloc[0]["admission_age"] == 52
    assert df.iloc[1]["admission_age"] == 76


def test_clean_data_requires_expected_columns():
    """
    Test that clean_data raises an error when required columns are missing.
    """
    processor = PatientDataProcessor()
    processor.merged_df = build_synthetic_merged_df().drop(columns=["hospital_expire_flag"])

    with pytest.raises(ValueError, match="missing required columns"):
        processor.clean_data()


def test_mortality_predictor_split_data_with_synthetic_input():
    """
    Test that train-test split works with synthetic modeling data.
    """
    processor = PatientDataProcessor()
    processor.merged_df = build_synthetic_merged_df()
    processor.clean_data()
    processor.feature_engineering()

    predictor = MortalityPredictor(processor)
    predictor.split_data(test_size=0.3, random_state=42)

    assert predictor.X_train is not None
    assert predictor.X_test is not None
    assert predictor.y_train is not None
    assert predictor.y_test is not None
    assert len(predictor.X_train) == 2
    assert len(predictor.X_test) == 2
    assert predictor.y_train.nunique() == 2
    assert predictor.y_test.nunique() == 2
    assert predictor.processor is processor


def test_train_logistic_regression_requires_split_first():
    """
    Test that training raises an exception if split_data has not been run.
    """
    predictor = MortalityPredictor(build_synthetic_model_df())

    with pytest.raises(RuntimeError, match="Run split_data\\(\\) first"):
        predictor.train_logistic_regression()


def test_predictor_requires_target_column():
    """
    Test that the predictor validates the presence of the mortality target.
    """
    df = build_synthetic_model_df().drop(columns=["mortality"])

    with pytest.raises(ValueError, match="mortality"):
        MortalityPredictor(df)
