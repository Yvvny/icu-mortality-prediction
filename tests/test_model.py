from modules.data_processor import PatientDataProcessor
from modules.model import MortalityPredictor


def test_data_processor_returns_required_columns():
    """
    Test whether processed model data contains the required columns.
    """
    processor = PatientDataProcessor("data")
    processor.load_data()
    processor.merge_data()
    processor.clean_data()
    processor.feature_engineering()

    df = processor.get_model_data()
    expected_columns = ["anchor_age", "gender_encoded", "los", "mortality"]

    assert list(df.columns) == expected_columns
    assert len(df) > 0


def test_mortality_predictor_split_data():
    """
    Test whether train-test split creates non-empty train and test sets.
    """
    processor = PatientDataProcessor("data")
    processor.load_data()
    processor.merge_data()
    processor.clean_data()
    processor.feature_engineering()

    df = processor.get_model_data()

    predictor = MortalityPredictor(df)
    predictor.split_data()

    assert predictor.X_train is not None
    assert predictor.X_test is not None
    assert predictor.y_train is not None
    assert predictor.y_test is not None
    assert len(predictor.X_train) > 0
    assert len(predictor.X_test) > 0