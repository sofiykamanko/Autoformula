import pandas as pd
import numpy as np

class FeatureTypeDetection:
    """
    Utilities for detecting and working with semantic feature types.
    """

    @staticmethod
    def detect_feature_types(df: pd.DataFrame, min_success: float = 0.9, detect_mixed: bool = False) -> pd.DataFrame:

        """
    Infer high-level semantic feature types for DataFrame columns.

    The function classifies each column into one of the following types:
    numeric, categorical, datetime, bool, or text. Detection is based on
    pandas dtypes, value distributions, cardinality, entropy, and
    string-to-numeric convertibility.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset for feature type inference.
    min_success : float, default=0.9
        Minimum fraction of successfully converted values required to
        treat an object column as numeric.
    detect_mixed : bool, default=False
        If True, flags columns containing a mixture of numeric-like and
        non-numeric values.

    Returns
    -------
    pandas.DataFrame
        Summary table with inferred feature types containing:
        - column : column name
        - pandas_dtype : original pandas dtype
        - detected_type : inferred semantic type
        - mixed_type : boolean flag (only if detect_mixed=True)

    Examples
    --------
    >>> detect_feature_types(df)
    >>> detect_feature_types(df, min_success=0.8)
    >>> detect_feature_types(df, detect_mixed=True)
    """

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
        Identify columns with conflicts between pandas dtypes and semantic types.
    
        Flags cases where the technical pandas dtype contradicts the inferred
        semantic meaning of a feature, such as object columns detected as
        non-text or numeric columns detected as categorical.
    
        Parameters
        ----------
        types_df : pandas.DataFrame
            Output of detect_feature_types(). Must contain the columns
            ['column', 'pandas_dtype', 'detected_type'].
    
        Returns
        -------
        pandas.DataFrame
            Subset of columns with dtype–semantic mismatches. Includes
            'mixed_type' if present in the input.
    
        Examples
        --------
        >>> types = detect_feature_types(df, detect_mixed=True)
        >>> type_mismatch(types)
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
    def handle_selected_missmatch(
        df: pd.DataFrame,
        types_df: pd.DataFrame,
        selected: list[str],
        numeric_threshold: float = 0.6
        ) -> pd.DataFrame:
        """
        Convert selected columns to their inferred semantic types.
    
        Applies type conversions only to specified columns based on
        detect_feature_types() output. Mixed-type columns are resolved
        using a numeric fraction threshold.
    
        Parameters
        ----------
        df : pandas.DataFrame
            Input dataset.
        types_df : pandas.DataFrame
            Output of detect_feature_types().
        selected : list[str]
            Columns to convert.
        numeric_threshold : float, default=0.6
            Minimum numeric fraction required to convert mixed-type columns
            to numeric; otherwise converted to categorical.
    
        Returns
        -------
        pandas.DataFrame
            Copy of the dataset with applied type conversions.
    
        Examples
        --------
        >>> types = detect_feature_types(df, detect_mixed=True)
        >>> df_fixed = handle_selected_missmatch(df, types, selected=["age", "gender"])
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

        """
        Compute basic per-feature summary statistics.
    
        Generates a compact overview of each column including uniqueness,
        missing values, dominance, and constancy, useful for quick
        dataset profiling and feature quality checks.
    
        Parameters
        ----------
        df : pandas.DataFrame
            Input dataset.
    
        Returns
        -------
        pandas.DataFrame
            Feature-level summary indexed by column name with:
            - dtype
            - n_unique
            - missing_rate
            - is_constant
            - dominant_ratio
            - dominant_value
    
        Examples
        --------
        >>> feature_summary(df)
        """

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

    
    
