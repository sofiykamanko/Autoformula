import pandas as pd
import numpy as np

class FeatureTypeDetection:
    """
    Utilities for detecting and working with semantic feature types.
    """

    @staticmethod
    def detect_feature_types(df: pd.DataFrame, min_success: float = 0.9, detect_mixed: bool = False) -> pd.DataFrame:

        '''
        Detects broad semantic type of each column:
            - numeric - integers / floats or string columns mostly convertible to numbers (default min_success=0.9)
            - categorical - discrete values with low unique count
            - datetime - pandas datetime dtype
            - bool - boolean dtype or simple Y/N, 0/1 encodings
            - text - high-cardinality free text

        Parameters
        ----------
        df : Input dataset for feature type detection (can be used cleaned df = output of preprocess_dataset() function )

        min_success : float, optional (default=0.9)
            Minimum proportion of successfully converted values (for string-to-numeric casting) required to treat an object
            column as numeric.

        detect_mixed : bool, optional (default=False)
            If True, flags columns that contain a mix of numeric-looking and text values.

        Returns
        -------
        pd.DataFrame
            A summary df with the following columns:
            - 'column' : column name
            - 'pandas_dtype' : technical dtype from pandas
            - 'detected_type' : inferred semantic type

        '''

        def find_mixed_type_columns(df):
            """Returns a list of column names where 10–90% of entries look numeric."""
            mixed = []
            for col in df.columns:
                s = df[col].dropna().astype(str)
                is_num = (
                    s
                    .str.replace('.', '', 1, regex=False)
                    .str.replace('-', '', 1, regex=False)
                    .str.isnumeric()
                )
                num_frac = is_num.mean()
                if 0.1 < num_frac < 0.9:
                    mixed.append(col)
            return mixed


        records = []

        for col in df.columns:
            s = df[col]
            pd_dtype = str(s.dtype)
            detected = None

            if np.issubdtype(s.dtype, np.datetime64):
                detected = "datetime"

            elif s.dtype == bool or s.dtype.name == "boolean":
                detected = "bool"

            else:
                if s.dtype == "object":
                    vals = s.dropna().astype(str).str.lower().unique()
                    if len(vals) <= 2 and set(vals).issubset({"y", "n", "yes", "no", "true", "false", "1", "0"}):
                        detected = "bool"

                if detected is None and np.issubdtype(s.dtype, np.number):
                    detected = "numeric"

                elif detected is None and s.dtype == "object":

                    s_num = pd.to_numeric(s, errors="coerce")
                    success = s_num.notna().mean()
                    if success >= min_success:
                        detected = "numeric"

                if detected is None:
                    s_str = s.dropna().astype(str)
                    n_unique = s_str.nunique()
                    n_total = len(s_str)
                    unique_ratio = n_unique / max(1, n_total)
                    avg_len = s_str.map(len).mean()

                    p = s_str.value_counts(normalize=True)
                    entropy = -np.sum(p * np.log2(p)) if len(p) > 0 else 0

                    if (
                        n_unique < 30
                        or (unique_ratio < 0.05 and avg_len < 10 and entropy < np.log2(50))
                    ):
                        detected = "categorical"
                    else:
                        detected = "text"


            records.append({
                "column": col,
                "pandas_dtype": pd_dtype,
                "detected_type": detected
            })

        result = pd.DataFrame(records)

        # Optional: mark mixed
        if detect_mixed:
            mixed_cols = find_mixed_type_columns(df)
            result["mixed_type"] = result["column"].isin(mixed_cols)

        return result


    @staticmethod
    def type_mismatch(types_df):
        """
    Return columns where pandas dtypes conflict with detected semantic types.

    Checks:
    - object dtype but semantic type ≠ 'text'
    - numeric dtype but semantic type == 'categorical'

    Parameters
    ----------
    types_df : DataFrame
        Output of detect_feature_types(), must contain
        ['column', 'pandas_dtype', 'detected_type'].

    Returns
    -------
    DataFrame
        Subset of columns with dtype–semantic mismatches (includes mixed_type if present).
    """


        required_cols = {"pandas_dtype", "detected_type"}

        if not required_cols.issubset(types_df.columns):

            raise ValueError(f"types_df must contain {required_cols}")

        mismatched = types_df[

            (types_df["pandas_dtype"].str.contains("object") & (types_df["detected_type"] != "text"))
            | ((types_df["pandas_dtype"].str.contains("float|int")) & (types_df["detected_type"] == "categorical"))

        ]

        display_cols = ["column", "pandas_dtype", "detected_type"]
        if "mixed_type" in types_df.columns:
            display_cols.append("mixed_type")

        return mismatched[display_cols]

    @staticmethod
    def handle_selected_columns(
        df: pd.DataFrame,
        types_df: pd.DataFrame,
        selected: list[str],
        numeric_threshold: float = 0.6
    ) -> pd.DataFrame:
        """
        Convert only selected columns to their semantic types, including mixed-type handling.

        Logic:
        - For mixed-type columns: if numeric_fraction > numeric_threshold - convert to numeric,
        else - convert to categorical.
        - For normal columns: convert based on detected_type (bool, categorical, numeric).

        Parameters
        ----------
        df : DataFrame — input dataset.
        types_df : DataFrame — output of detect_feature_types().
        selected : list[str] — columns to convert.
        numeric_threshold : float — cutoff for mixed-type numeric detection.

        Returns
        -------
        DataFrame — copy of df with applied conversions.
        """


        df_out = df.copy()

        type_map = types_df.set_index("column")["detected_type"].to_dict()
        mixed_map = types_df.set_index("column")["mixed_type"].to_dict() if "mixed_type" in types_df.columns else {}

        for col in selected:

            if col not in df_out.columns:

                print(f"Column '{col}' not found, skipping.")
                continue

            target_type = type_map.get(col)
            is_mixed = mixed_map.get(col, False)

            # handle mixed-type columns
            if is_mixed:
                s = df_out[col].dropna().astype(str)
                is_num = (
                    s
                    .str.replace('.', '', 1, regex=False)
                    .str.replace('-', '', 1, regex=False)
                    .str.isnumeric()
                )
                num_frac = is_num.mean()

                if num_frac > numeric_threshold:
                    df_out[col] = pd.to_numeric(df_out[col], errors="coerce")
                    print(f"'{col}' marked as mixed-type - converted to numeric (≈{num_frac:.0%} numeric).")
                else:
                    df_out[col] = df_out[col].astype("category")
                    print(f"'{col}' marked as mixed-type - converted to categorical (≈{num_frac:.0%} numeric).")
                continue

            # normal conversion flow
            try:
                if target_type == "bool":
                    df_out[col] = (
                        df_out[col].astype(str).str.lower()
                        .map({"y": True, "yes": True, "1": True, "n": False, "no": False, "0": False})
                        .astype("boolean")
                    )
                    print(f"Converted '{col}' to boolean.")

                elif target_type == "categorical":
                    df_out[col] = df_out[col].astype("category")
                    print(f"Converted '{col}' to category.")

                elif target_type == "numeric":
                    df_out[col] = pd.to_numeric(df_out[col], errors="coerce")
                    print(f"Converted '{col}' to numeric.")

                else:
                    print(f"Skipping '{col}' ({target_type}) — not a convertible type.")

            except Exception as e:
                print(f"Error converting '{col}': {e}")

        return df_out
    
    @staticmethod
    def feature_summary(df: pd.DataFrame) -> pd.DataFrame:

        '''
        Returns a DataFrame indexed by feature names with summary statistics.

        Parameters
        ----------
        df: Input dataset for feature summery (can be used df_out = output of handle_selected_columns() function).

        Returns
        -------
        pd.DataFrame
            Summary df with:
            - dtype : pandas data type of the feature
            - n_unique : number of unique non-null values
            - missing_rate : fraction of missing values
            - is_constant : True if all values are identical
            - dominant_ratio : share of the most frequent value
            - dominant_value : the most frequent value itself

        '''

        if df.empty:
            return pd.DataFrame(columns=["n_unique", "missing_rate", "is_constant", "dominant_ratio", "dominant_value"])

        n_rows = len(df)
        rows = []

        for col in df.columns:
            series = df[col]
            counts = series.value_counts(dropna=True)

            n_unique = series.nunique(dropna=True)
            missing_rate = round(series.isna().mean(), 4)
            dominant_value = counts.idxmax() if not counts.empty else None
            dominant_ratio = round(counts.max() / n_rows, 4) if not counts.empty else None

            rows.append({
                "feature": col,
                "dtype": str(series.dtype),
                "n_unique": n_unique,
                "missing_rate": missing_rate,
                "is_constant": (n_unique == 1),
                "dominant_ratio": dominant_ratio,
                "dominant_value": dominant_value
            })

        return pd.DataFrame(rows).set_index("feature")

