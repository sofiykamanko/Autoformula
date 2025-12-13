import pandas as pd
from sklearn.model_selection import train_test_split

class SanitizationEngine:

    def __init__(
        self,
        target=None,
        sample_rows=None,
        stratify=True,
        random_state=42
    ):
        self.target = target
        self.sample_rows = sample_rows
        self.stratify = stratify
        self.random_state = random_state

    def preprocess_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        '''

        Remove duplicate rows/columns and optionally sample the dataset.

        1. Drop duplicate rows.
        2. Drop duplicate columns (identical across all rows).
        3. Optionally sample `sample_rows` rows:
        - Stratified by `target` if enabled and possible,
        - Otherwise random.

        Parameters
        ----------
        df : DataFrame
        target : str or None — column for stratified sampling
        sample_rows : int or None — number of rows to sample
        stratify : bool — preserve class proportions when sampling
        random_state : int — seed

        Returns
        -------
        DataFrame — cleaned (and optionally sampled) dataset.
        '''

        print(f"Initial dataset shape: {df.shape}")

        dup_rows = df.duplicated().sum()
        if dup_rows > 0:
            print(f"Found {dup_rows} duplicate rows - removing them.")
            df = df.drop_duplicates()
        else:
            print("No duplicate rows found.")

        dup_cols = df.T.duplicated().sum()
        if dup_cols > 0:
            print(f"Found {dup_cols} duplicate columns — removing them.")
            df = df.loc[:, ~df.T.duplicated()]
        else:
            print("No duplicate columns found.")

        if self.sample_rows is not None and self.sample_rows < len(df):
            if (
                self.target is not None
                and self.stratify
                and df[self.target].nunique() > 1
            ):
                df, _ = train_test_split(
                    df,
                    train_size=self.sample_rows,
                    stratify=df[self.target],
                    random_state=self.random_state
                )
            else:
                df = df.sample(
                    n=self.sample_rows,
                    random_state=self.random_state
                )

            df = df.reset_index(drop=True)
        else:
            print("Sampling not applied (using full dataset).")

        print(f"Dataset shape after sanitization: {df.shape}")
        return df
