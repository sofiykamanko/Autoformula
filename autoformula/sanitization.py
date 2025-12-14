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
        """
        Clean a dataset by removing duplicate rows and columns, with optional sampling.
    
        This function performs basic dataset sanitization steps that are commonly
        required before feature analysis or modeling:
    
        1. Removes fully duplicated rows.
        2. Removes duplicated columns (columns with identical values across all rows).
        3. Optionally samples a fixed number of rows, using stratified sampling
           if a target column is provided.
    
        Parameters
        ----------
        df : pandas.DataFrame
            Input dataset to be sanitized.
    
        target : str or None, default=None
            Name of the target column used for stratified sampling.
            If None, random sampling is applied.
    
        sample_rows : int or None, default=None
            Number of rows to sample from the dataset.
            If None or greater than the dataset size, no sampling is performed.
    
        stratify : bool, default=True
            Whether to preserve class proportions during sampling.
            Applied only if `target` is provided and has more than one unique value.
    
        random_state : int, default=42
            Random seed used for reproducible sampling.
    
        Returns
        -------
        pandas.DataFrame
            A cleaned copy of the dataset with duplicates removed and
            optional sampling applied.
    
        Examples
        --------
        Basic usage without sampling:
    
        >>> df_clean = preprocess_dataset(df)
    
        Stratified sampling with a target column:
    
        >>> df_clean = preprocess_dataset(
        ...     df,
        ...     target="label",
        ...     sample_rows=5000
        ... )
    
        Random sampling without stratification:
    
        >>> df_clean = preprocess_dataset(
        ...     df,
        ...     sample_rows=1000,
        ...     stratify=False
        ... )
        """
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
