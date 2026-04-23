from pathlib import Path

import pandas as pd
import random


class PatientDataProcessor:
    """
    Class to load, merge, clean, and prepare ICU patient data.
    """

    def __init__(self, data_dir="data"):
        """
        Initialize the data processor.

        Args:
            data_dir (str): Folder containing patients.csv, admissions.csv, and icustays.csv
        """
        self.data_dir = Path(data_dir)
        self.patients_df = None
        self.admissions_df = None
        self.icustays_df = None
        self.merged_df = None

    def _require_merged_df(self, message):
        """
        Return the current merged dataframe or raise a runtime error.
        """
        if self.merged_df is None:
            raise RuntimeError(message)
        return self.merged_df

    def __str__(self):
        """
        Return a readable summary of the processed dataset.
        """
        if self.merged_df is None:
            return "Data not processed yet"
        return f"Dataset with {len(self.merged_df)} records"

    def __len__(self):
        """
        Return number of rows in final processed dataframe.
        """
        if self.merged_df is None:
            return 0
        return len(self.merged_df)

    def load_data(self):
        """
        Load the three CSV files.

        Raises:
            Exception: If any file cannot be loaded
        """
        file_map = {
            "patients_df": self.data_dir / "patients.csv",
            "admissions_df": self.data_dir / "admissions.csv",
            "icustays_df": self.data_dir / "icustays.csv",
        }

        try:
            for attr_name, file_path in file_map.items():
                setattr(self, attr_name, pd.read_csv(file_path))
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Missing required data file: {exc.filename}") from exc
        except Exception as exc:
            raise RuntimeError(f"Error loading files: {exc}") from exc

    def merge_data(self):
        """
        Merge patients, admissions, and icustays data.

        Returns:
            pd.DataFrame: merged dataframe

        Raises:
            Exception: If data has not been loaded
        """
        if self.patients_df is None or self.admissions_df is None or self.icustays_df is None:
            raise RuntimeError("Run load_data() first")

        df = pd.merge(self.patients_df, self.admissions_df, on="subject_id", how="inner")
        df = pd.merge(df, self.icustays_df, on=["subject_id", "hadm_id"], how="inner")

        self.merged_df = df
        return df

    def clean_data(self):
        """
        Clean merged dataset.

        Returns:
            pd.DataFrame: cleaned dataframe

        Raises:
            Exception: If merged data is not available
        """
        df = self._require_merged_df("Run merge_data() first").copy()

        date_cols = ["admittime", "dischtime", "deathtime", "intime", "outtime", "dod"]
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        required_cols = ["anchor_age", "gender", "los", "hospital_expire_flag", "first_careunit"]
        missing_required_cols = [col for col in required_cols if col not in df.columns]
        if missing_required_cols:
            missing_cols = ", ".join(missing_required_cols)
            raise ValueError(f"Merged data is missing required columns: {missing_cols}")

        df = df.dropna(subset=required_cols)

        df = df[df["hospital_expire_flag"].isin([0, 1])]

        self.merged_df = df
        return df

    def feature_engineering(self):
        """
        Create additional features for analysis and modeling.

        Returns:
            pd.DataFrame: dataframe with engineered features

        Raises:
            Exception: If cleaned data is not available
        """
        df = self._require_merged_df("Run clean_data() first").copy()

        age_bins = (0, 20, 40, 60, 80, 120)  # tuple = immutable type
        age_labels = ["0-20", "21-40", "41-60", "61-80", "80+"]

        if {"anchor_age", "anchor_year", "admittime"}.issubset(df.columns):
            admit_year = df["admittime"].dt.year
            df["admission_age"] = df["anchor_age"] + (admit_year - df["anchor_year"])
        else:
            df["admission_age"] = df["anchor_age"]

        df["age_group"] = pd.cut(
            df["admission_age"],
            bins=age_bins,
            labels=age_labels
        )

        df["gender_encoded"] = df["gender"].apply(lambda x: 1 if str(x).upper() == "M" else 0)
        df["mortality"] = df["hospital_expire_flag"].astype(int)

        self.merged_df = df
        return df

    def get_model_data(self):
        """
        Return the modeling dataframe.

        Returns:
            pd.DataFrame: subset of columns for machine learning

        Raises:
            Exception: If feature engineering has not been run
        """
        df = self._require_merged_df("Run feature_engineering() first")
        required_model_cols = ["admission_age", "gender_encoded", "los", "mortality"]
        missing_model_cols = [col for col in required_model_cols if col not in df.columns]
        if missing_model_cols:
            missing_cols = ", ".join(missing_model_cols)
            raise RuntimeError(f"Run feature_engineering() first. Missing columns: {missing_cols}")

        return df[required_model_cols].copy()

    def get_unique_units(self):
        """
        Return unique ICU care units using set operation and list comprehension.

        Returns:
            list: sorted unique ICU unit names
        """
        df = self._require_merged_df("Run feature_engineering() first")
        units = sorted(list(set([u for u in df["first_careunit"].dropna()])))
        return units

    def row_generator(self):
        """
        Generator function to yield rows one by one.

        Yields:
            tuple: index and row
        """
        df = self._require_merged_df("Run feature_engineering() first")
        for idx, row in df.iterrows():
            yield idx, row

    def get_random_sample(self, n=5, random_state=42):
        """
        Return a random sample of rows using built-in random module.

        Args:
            n (int): number of rows to sample

        Returns:
            pd.DataFrame: sampled rows
        """
        df = self._require_merged_df("Run feature_engineering() first")
        n = min(n, len(df))
        sampler = random.Random(random_state)
        indices = sampler.sample(list(df.index), n)
        return df.loc[indices]
