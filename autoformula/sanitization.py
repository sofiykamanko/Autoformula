import pandas as pd
from sklearn.model_selection import train_test_split

class SanitizationEngine:

    @staticmethod
    def preprocess_dataset(
        df: pd.DataFrame,
        target: str | None = None,
        sample_rows: int | None = None,
        stratify: bool = True,
        random_state: int = 42
    ) -> pd.DataFrame:
        
    """
    Clean a dataset by removing duplicate rows/columns and optionally sampling rows.

    The function removes fully duplicated rows, drops duplicated columns
    (columns with identical values), and optionally samples a fixed number
    of rows with support for stratified sampling by a target column.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset.
    target : str or None, default=None
        Target column used for stratified sampling.
    sample_rows : int or None, default=None
        Number of rows to sample; sampling is skipped if None or too large.
    stratify : bool, default=True
        Preserve class proportions during sampling if possible.
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    pandas.DataFrame
        Preprocessed dataset with duplicates removed and optional sampling applied.

    Examples
    --------
    >>> preprocess_dataset(df)
    >>> preprocess_dataset(df, target="label", sample_rows=1000)
    >>> preprocess_dataset(df, sample_rows=500, stratify=False)
    """

        print(f"Initial dataset shape: {df.shape}")

    # duplicate rows
        dup_rows = df.duplicated().sum()

        if dup_rows > 0:
            print(f"Found {dup_rows} duplicate rows - removing them.")
            df = df.drop_duplicates()
        else:
            print("No duplicate rows found.")

    # duplicate columns
        dup_cols = df.T.duplicated().sum()

        if dup_cols > 0:
            print(f"Found {dup_cols} duplicate columns — removing them.")
            df = df.loc[:, ~df.T.duplicated()]
        else:
            print("No duplicate columns found.")

        # stratified sampling

        if sample_rows is not None and sample_rows < len(df):

            if target is not None and stratify and df[target].nunique() > 1:

                df_sampled, _ = train_test_split(
                    df,
                    train_size=sample_rows,
                    stratify=df[target],
                    random_state=random_state
                )
            else:
                
                df_sampled = df.sample(n=sample_rows, random_state=random_state)

            df = df_sampled.reset_index(drop=True)
        else:
            print("Sampling not applied (using full dataset).")


        print(f"Dataset shape after sanitization: {df.shape}")

        return df

