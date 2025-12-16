import numpy as np
import pandas as pd
from itertools import combinations
from pandas.api.types import is_numeric_dtype


class FeatureMixing:
    """
    Feature generation via unary and binary transformations.
    """

    @staticmethod
    def generate_feature_combinations(
        df: pd.DataFrame,
        max_pairs: int = 50,
        epsilon: float = 1e-6,
        random_state: int = 42
    ) -> pd.DataFrame:
        
        """
        Generate non-linear and interaction-based numeric feature combinations.
    
        Creates common unary transformations (log, sqrt, square, scaling, ranking)
        and pairwise interactions (products, ratios, differences) for numeric features.
        Binary interactions are randomly sampled to limit feature explosion.
    
        Parameters
        ----------
        df : pandas.DataFrame
            Input dataset.
        max_pairs : int, default=50
            Maximum number of feature pairs used for binary interactions.
        epsilon : float, default=1e-6
            Small constant for numerical stability.
        random_state : int, default=42
            Random seed controlling pair sampling.
    
        Returns
        -------
        pandas.DataFrame
            Dataset extended with generated feature combinations.
    
        Examples
        --------
        >>> df_aug = generate_feature_combinations(df)
        """
        
        np.random.seed(random_state)

        df_base = df.copy()
        new_features = {}

        num_features = [c for c in df.columns if is_numeric_dtype(df[c])]

        # unary 
        for col in num_features:
            x = df[col]

            new_features[f"log_{col}"] = np.log(x + 1)
            new_features[f"sqrt_{col}"] = np.sqrt(np.clip(x, 0, None))
            new_features[f"{col}_sq"] = x ** 2
            new_features[f"{col}_scaled"] = x / (x.std() + epsilon)
            new_features[f"{col}_rank"] = x.rank(pct=True)
            new_features[f"{col}_dev_median"] = (x - x.median()).abs()

        # binary 
        all_pairs = list(combinations(num_features, 2))
        np.random.shuffle(all_pairs)

        for a, b in all_pairs[:max_pairs]:
            x1, x2 = df[a], df[b]

            new_features[f"{a}_mul_{b}"] = x1 * x2
            new_features[f"{a}_div_{b}"] = x1 / (x2 + epsilon)
            new_features[f"abs_{a}_minus_{b}"] = (x1 - x2).abs()
            new_features[f"log_{a}_mul_{b}"] = np.log(x1 * x2 + 1)
            new_features[f"{a}_div_sqrt_{b}"] = x1 / (
                np.sqrt(np.clip(x2, 0, None)) + epsilon
            )

        return pd.concat([df_base, pd.DataFrame(new_features)], axis=1)

    
    @staticmethod
    def generate_custom_feature_combinations(
        df: pd.DataFrame,
        unary_ops: dict[str, callable] | None = None,
        binary_ops: dict[str, callable] | None = None,
        max_pairs: int = 50,
        random_state: int = 42
    ) -> pd.DataFrame:
        """
        Generate numeric feature combinations using custom user-defined operations.
    
        Applies arbitrary unary transformations f(x) and binary interactions f(x1, x2)
        to numeric features. Binary feature pairs are randomly sampled to control
        feature explosion. Invalid operations are safely skipped.
    
        Parameters
        ----------
        df : pandas.DataFrame
            Input dataset.
        unary_ops : dict[str, callable] or None, default=None
            Mapping of operation names to unary functions f(x).
        binary_ops : dict[str, callable] or None, default=None
            Mapping of operation names to binary functions f(x1, x2).
        max_pairs : int, default=50
            Maximum number of feature pairs used for binary interactions.
        random_state : int, default=42
            Random seed controlling pair sampling.
    
        Returns
        -------
        pandas.DataFrame
            Dataset extended with generated custom feature combinations.
    
        Examples
        --------
        >>> unary_ops = {"log": lambda x: np.log(x + 1)}
        >>> binary_ops = {"ratio": lambda x1, x2: x1 / (x2 + 1e-6)}
        >>> generate_custom_feature_combinations(df, unary_ops, binary_ops)
        """
        
        np.random.seed(random_state)

        df_base = df.copy()
        new_features = {}

        num_features = [c for c in df.columns if is_numeric_dtype(df[c])]

        #unary 
        if unary_ops:
            for col in num_features:
                x = df[col]
                for name, func in unary_ops.items():
                    try:
                        new_features[f"{name}_{col}"] = func(x)
                    except Exception:
                        continue  # safe skip

        #binary 
        if binary_ops:
            pairs = list(combinations(num_features, 2))
            np.random.shuffle(pairs)

            for a, b in pairs[:max_pairs]:
                x1, x2 = df[a], df[b]
                for name, func in binary_ops.items():
                    try:
                        new_features[f"{a}_{name}_{b}"] = func(x1, x2)
                    except Exception:
                        continue

        return pd.concat([df_base, pd.DataFrame(new_features)], axis=1)
