import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import (
    spearmanr,
    pearsonr,
    kendalltau,
    chi2_contingency,
    kruskal
)
from pandas.api.types import is_numeric_dtype


class FeatureAnalysis:
    """
    Statistical analysis utilities for feature evaluation.
    """

    @staticmethod
    def check_correlations(df_num):
        """
        Compute correlation matrices for numeric features.
    
        Calculates Pearson, Spearman, and Kendall correlation matrices
        using numeric and boolean columns only.
    
        Parameters
        ----------
        df_num : pandas.DataFrame
            Input dataset.
    
        Returns
        -------
        dict[str, pandas.DataFrame]
            Correlation matrices with keys: 'pearson', 'spearman', 'kendall'.
    
        Examples
        --------
        >>> corrs = check_correlations(df)
        >>> corrs["spearman"]
        """

        numeric_df = df_num.select_dtypes(include=["float64", "int64", "bool"])

        pearson_corr = numeric_df.corr(method="pearson")
        spearman_corr = numeric_df.corr(method="spearman")
        kendall_corr = numeric_df.corr(method="kendall")

        print("Pearson:\n", pearson_corr.head())
        print("\nSpearman:\n", spearman_corr.head())
        print("\nKendall:\n", kendall_corr.head())

        return {
            "pearson": pearson_corr,
            "spearman": spearman_corr,
            "kendall": kendall_corr
        }

    
    @staticmethod
    def plot_correlation_heatmaps(corrs_dict, figsize=(30, 10), annot_size=12):
        """
        Visualize correlation matrices as heatmaps.
    
        Plots Pearson, Spearman, and Kendall correlation matrices returned
        by check_correlations() using seaborn heatmaps.
    
        Parameters
        ----------
        corrs_dict : dict[str, pandas.DataFrame]
            Correlation matrices with keys 'pearson', 'spearman', and 'kendall'.
        figsize : tuple, default=(30, 10)
            Figure size.
        annot_size : int, default=12
            Font size for correlation value annotations.
    
        Returns
        -------
        None
            Displays the heatmaps.
        
        Examples
        --------
        >>> corrs = check_correlations(df)
        >>> plot_correlation_heatmaps(corrs)
        """

        corr_matrices = [
        ("Pearson Correlation", corrs_dict.get("pearson")),
        ("Spearman Correlation", corrs_dict.get("spearman")),
        ("Kendall Correlation", corrs_dict.get("kendall")),
        ]

        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=figsize)

        heatmap_kwargs = dict(
            annot=True,
            fmt=".2f",
            cmap="RdBu",
            center=0,
            annot_kws={"size": annot_size},
            cbar=True
        )

        for ax, (title, corr) in zip(axes, corr_matrices):
            sns.heatmap(corr, ax=ax, **heatmap_kwargs)
            ax.set_title(title, fontsize=14)

        plt.tight_layout()
        plt.show()

    
    @staticmethod
    def cramers_v_matrix(df: pd.DataFrame, show_plot: bool = True):
        """
        Compute a Cramér’s V association matrix for categorical features.
    
        Calculates pairwise Cramér’s V statistics for categorical and boolean
        columns to measure the strength of association between features.
        Optionally visualizes the result as a heatmap.
    
        Parameters
        ----------
        df : pandas.DataFrame
            Input dataset; only categorical and boolean columns are used.
        show_plot : bool, default=True
            Whether to display the association heatmap.
    
        Returns
        -------
        pandas.DataFrame
            Symmetric matrix of Cramér’s V values.
    
        Examples
        --------
        >>> V = cramers_v_matrix(df)
        >>> V.loc["gender", "country"]
        """

        df_cat = df.select_dtypes(include=["object", "category", "bool"]).copy()

        cols = df_cat.columns
        V = pd.DataFrame(index=cols, columns=cols, dtype=float)

        for a in cols:
            for b in cols:
                ct = pd.crosstab(df_cat[a], df_cat[b])
                if ct.shape[0] < 2 or ct.shape[1] < 2:
                    V.loc[a, b] = np.nan
                else:
                    chi2, _, _, _ = chi2_contingency(ct, correction=False)
                    n = ct.values.sum()
                    V.loc[a, b] = np.sqrt(chi2 / (n * (min(ct.shape) - 1)))

        print(f"Cramérs V computed for {len(cols)} categorical features.")

        if show_plot and len(cols) > 1:
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(
                V,
                ax=ax,
                annot=True,
                fmt=".2f",
                cmap="RdBu",
                center=0.5,
                linewidths=0.5
            )
            ax.set_title("Cramér’s V Association Heatmap", fontsize=14)
            plt.tight_layout()
            plt.show()

        return V


    @staticmethod
    def compute_overlap(
        df: pd.DataFrame,
        top_k: int = 3,
        use_pearson: bool = False,
        use_kendall: bool = False,
        return_pairwise: bool = True
    ):
        :
        """
        Compute feature overlap scores to estimate redundancy between features.
    
        Measures pairwise feature overlap using:
        - numeric–numeric: |Spearman| (optionally Pearson / Kendall)
        - categorical–categorical: Cramér’s V
        - numeric–categorical: correlation ratio (η²)
    
        Per-feature overlap is defined as the mean of the top-k strongest
        pairwise overlaps; higher values indicate higher redundancy.
    
        Parameters
        ----------
        df : pandas.DataFrame
            Dataset containing feature columns only.
        top_k : int, default=3
            Number of strongest overlaps aggregated per feature.
        use_pearson : bool, default=False
            Include Pearson correlation for numeric-numeric pairs.
        use_kendall : bool, default=False
            Include Kendall tau for numeric-numeric pairs.
        return_pairwise : bool, default=True
            Whether to return the full pairwise overlap table.
    
        Returns
        -------
        dict
            {
                "feature_overlap": pandas.DataFrame,
                "pairwise_overlap": pandas.DataFrame | None
            }
    
        Examples
        --------
        >>> res = compute_overlap(df)
        >>> res["feature_overlap"].head()
        """
        # helpers

        def cramers_v(x, y):

            table = pd.crosstab(x, y)
            if table.shape[0] < 2 or table.shape[1] < 2:
                return 0.0

            chi2 = chi2_contingency(table, correction=False)[0]

            n = table.values.sum()
            k = min(table.shape)
            v = np.sqrt(chi2 / (n * (k - 1)))

            return v


        def correlation_ratio(categories, values):

            categories = pd.Categorical(categories)
            values = np.asarray(values, dtype=float)

            overall_mean = np.nanmean(values)

            explained_variance = 0.0

            for category in categories.categories:
                group_values = values[categories == category]

                if len(group_values) == 0:
                    continue

                group_mean = np.nanmean(group_values)
                explained_variance += len(group_values) * (group_mean - overall_mean) ** 2

            total_variance = np.nansum((values - overall_mean) ** 2)

            if total_variance == 0:
                return 0.0

            return explained_variance / total_variance


        # pairwise overlap 

        features = df.columns
        rows = []

        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                f1 = features[i]
                f2 = features[j]

                x = df[f1]
                y = df[f2]

                x_is_num = pd.api.types.is_numeric_dtype(x)
                y_is_num = pd.api.types.is_numeric_dtype(y)

                if x_is_num and y_is_num:
                    x_num = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
                    y_num = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)

                    if np.nanstd(x_num) == 0 or np.nanstd(y_num) == 0:
                        overlap = 0.0
                    else:
                        overlap = abs(spearmanr(x_num, y_num, nan_policy="omit")[0])
                    if use_pearson:
                        overlap = max(overlap, abs(pearsonr(x, y)[0]))
                    if use_kendall:
                        overlap = max(overlap, abs(kendalltau(x, y)[0]))
                    pair_type = "numeric-numeric"
                    
                elif not x_is_num and not y_is_num:
                    overlap = cramers_v(x, y)
                    pair_type = "categorical-categorical"
                    
                else:
                    if x_is_num:
                        overlap = correlation_ratio(y, x)
                    else:
                        overlap = correlation_ratio(x, y)
                    pair_type = "numeric-categorical"

                
                rows.append({
                "feature_a": f1,
                "feature_b": f2,
                "type_pair": pair_type,
                "overlap_score": overlap
            })

        
        pairwise_df = pd.DataFrame(rows)

        feature_rows = []
        for feature in features:
            scores_as_a = pairwise_df.loc[pairwise_df["feature_a"] == feature, "overlap_score"]
            scores_as_b = pairwise_df.loc[pairwise_df["feature_b"] == feature, "overlap_score"]
            
            all_scores = pd.concat([scores_as_a, scores_as_b])
            
            strongest_scores = all_scores.sort_values(ascending=False).head(top_k)

            if len(strongest_scores) == 0:
                overlap_score = 0.0
            else:
                overlap_score = strongest_scores.mean()

            feature_rows.append({
                "feature": feature,
                "overlap_score": overlap_score
            })

        # feature-level table
        feature_df = pd.DataFrame(feature_rows)
        feature_df = feature_df.sort_values("overlap_score", ascending=True)
        feature_df = feature_df.reset_index(drop=True)

        return {
            "feature_overlap": feature_df,
            "pairwise_overlap": pairwise_df if return_pairwise else None
        }



    @staticmethod
    def compute_separability(
        X: pd.DataFrame,
        y: pd.Series,
        run_permutation: bool = True,
        n_permutations: int = 300
    ):
        
) -> pd.DataFrame:
    """
    Compute feature–target separability scores with optional permutation testing.

    For each feature, selects an association measure based on feature/target types:
    - numeric–numeric: |Spearman|
    - categorical–numeric: correlation ratio (η²)
    - categorical–categorical: Cramér’s V

    Optionally estimates statistical significance via a permutation test.

    Parameters
    ----------
    X : pandas.DataFrame
        Feature matrix.
    y : pandas.Series
        Target variable.
    run_permutation : bool, default=True
        Whether to compute permutation-based p-values.
    n_permutations : int, default=300
        Number of permutations used for p-value estimation.

    Returns
    -------
    pandas.DataFrame
        Ranked feature table with:
        - feature
        - separability_score
        - perm_p_value (if run_permutation=True)

    Examples
    --------
    >>> compute_separability(X, y)
    >>> compute_separability(X, y, run_permutation=False)
    """
    
        # helpers 

        def cramers_v(x, y):
            table = pd.crosstab(x, y)
            if table.shape[0] < 2 or table.shape[1] < 2:
                return 0.0
            chi2 = chi2_contingency(table, correction=False)[0]
            n = table.values.sum()
            k = min(table.shape)
            return np.sqrt(chi2 / (n * (k - 1)))

        def correlation_ratio(categories, values):
            categories = pd.Categorical(categories)
            values = np.asarray(values, dtype=float)

            mean_total = np.nanmean(values)
            ss_between = 0.0

            for cat in categories.categories:
                group = values[categories == cat]
                if len(group) == 0:
                    continue
                ss_between += len(group) * (np.nanmean(group) - mean_total) ** 2

            ss_total = np.nansum((values - mean_total) ** 2)
            return ss_between / ss_total if ss_total > 0 else 0.0

        def separability_single_feature(x, y):
            x_is_num = is_numeric_dtype(x)
            y_is_num = is_numeric_dtype(y)

            if y_is_num:
                if x_is_num:
                    return abs(spearmanr(x, y, nan_policy="omit")[0])
                else:
                    return correlation_ratio(x, y)
            else:
                if x_is_num:
                    return correlation_ratio(y, x)
                else:
                    return cramers_v(x, y)

        # main separability 

        rows = []

        for feature in X.columns:
            x = X[feature]
            score = separability_single_feature(x, y)

            rows.append({
                "feature": feature,
                "separability_score": score
            })

        df = pd.DataFrame(rows).sort_values(
            "separability_score", ascending=False
        ).reset_index(drop=True)

        # permutation test 

        if run_permutation:
            p_values = []

            for feature in df["feature"]:
                x = X[feature]
                observed = df.loc[df["feature"] == feature, "separability_score"].iloc[0]
                perm_scores = []

                for i in range(n_permutations):
                    y_perm = np.random.permutation(y)
                    perm_scores.append(separability_single_feature(x, y_perm))

                perm_scores = np.array(perm_scores)
                p_values.append(np.mean(perm_scores >= observed))

            df["perm_p_value"] = p_values

        return df


    @staticmethod
    def compute_impact(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Estimate statistical impact (effect size) of each feature on the target.
    
        Automatically selects an appropriate effect size metric and statistical
        test based on feature–target types:
        - numeric–numeric: |Spearman| correlation
        - numeric–categorical or categorical–numeric: η² + Kruskal–Wallis test
        - categorical–categorical: Cramér’s V + chi-square test
    
        Parameters
        ----------
        X : pandas.DataFrame
            Feature matrix.
        y : pandas.Series
            Target variable.
    
        Returns
        -------
        pandas.DataFrame
            Feature-level impact table sorted by decreasing impact with:
            - feature
            - impact_score
            - impact_metric
            - test
            - p_value
    
        Examples
        --------
        >>> compute_impact(X, y)
        """

        results = []

        # вrop NA target values (for variance-based statistics)
        mask = y.notna()
        X = X.loc[mask]
        y = y.loc[mask]

        for feature in X.columns:
            x = X[feature]

            x_is_num = is_numeric_dtype(x)
            y_is_num = is_numeric_dtype(y)

            # numeric + numeric 
            if x_is_num and y_is_num:
                corr, p = spearmanr(x, y, nan_policy="omit")

                impact = abs(corr)
                metric = "spearman"
                test = "spearman"

            # numeric + categorical 
            elif x_is_num and not y_is_num:
                categories = pd.Categorical(y)
                values = x.astype(float)
                values = values.dropna()

                mean_total = values.mean()
                ss_between = 0.0

                for cat in categories.categories:
                    group = values[y == cat]
                    ss_between += len(group) * (group.mean() - mean_total) ** 2

                ss_total = ((values - mean_total) ** 2).sum()
                impact = ss_between / ss_total if ss_total > 0 else 0.0

                groups = []

                for cat in categories.categories:
                    group = x[y == cat].dropna()
                    if len(group) > 0:
                        groups.append(group)

                p = kruskal(*groups).pvalue if len(groups) > 1 else np.nan


                metric = "eta_squared"
                test = "kruskal"

            # categorical + numeric 
            elif not x_is_num and y_is_num:
                categories = pd.Categorical(x)
                values = y.astype(float)
                values = values.dropna()

                mean_total = values.mean()
                ss_between = 0.0

                for cat in categories.categories:
                    group = values[x == cat]
                    ss_between += len(group) * (group.mean() - mean_total) ** 2

                ss_total = ((values - mean_total) ** 2).sum()
                impact = ss_between / ss_total if ss_total > 0 else 0.0

                groups = []

                for cat in categories.categories:
                    group = y[x == cat].dropna()
                    if len(group) > 0:
                        groups.append(group)

                p = kruskal(*groups).pvalue if len(groups) > 1 else np.nan


                metric = "eta_squared"
                test = "kruskal"

            # categorical + categorical 
            else:
                table = pd.crosstab(x, y)

                if table.shape[0] < 2 or table.shape[1] < 2:
                    impact = 0.0
                    p = np.nan
                else:
                    chi2, p, _, _ = chi2_contingency(table)
                    n = table.values.sum()
                    k = min(table.shape)
                    impact = np.sqrt(chi2 / (n * (k - 1)))

                metric = "cramers_v"
                test = "chi2"

            results.append({
                "feature": feature,
                "impact_score": impact,
                "impact_metric": metric,
                "test": test,
                "p_value": p
            })

        return (
            pd.DataFrame(results)
            .sort_values("impact_score", ascending=False)
            .reset_index(drop=True)
        )

    @staticmethod
    def compute_feature_quality(
        X: pd.DataFrame,
        y: pd.Series,
        weights=None,
        top_k_overlap: int = 3
    ):
        
) -> pd.DataFrame:
        """
        Compute a composite feature quality score.
    
        Ranks features by combining three complementary signals:
        - separability: strength of association with the target,
        - impact: statistical effect size,
        - overlap: redundancy with other features.
    
        The final score favors features with high signal and low redundancy.
    
        Parameters
        ----------
        X : pandas.DataFrame
            Feature matrix.
        y : pandas.Series
            Target variable.
        weights : dict or None, default=None
            Exponents controlling the contribution of each component:
            {'separability', 'impact', 'overlap'}.
        top_k_overlap : int, default=3
            Number of strongest overlaps used to estimate feature redundancy.
    
        Returns
        -------
        pandas.DataFrame
            Features ranked by decreasing quality with:
            - separability_score
            - impact_score
            - overlap_score
            - final_score
    
        Examples
        --------
        >>> compute_feature_quality(X, y)
        >>> compute_feature_quality(X, y, weights={"separability": 2, "impact": 1, "overlap": 1})
        """

        if weights is None:
            weights = {
                "separability": 1.0,
                "impact": 1.0,
                "overlap": 1.0
            }

        overlap_df = FeatureAnalysis.compute_overlap(
            df=X,
            top_k=top_k_overlap,
            return_pairwise=False
        )["feature_overlap"]

        separability_df = FeatureAnalysis.compute_separability(X, y)
        impact_df = FeatureAnalysis.compute_impact(X, y)

        df = (
            overlap_df
            .merge(separability_df, on="feature")
            .merge(impact_df, on="feature")
        )

        df["final_score"] = (
            df["separability_score"] ** weights["separability"]
            * df["impact_score"] ** weights["impact"]
            * (1 - df["overlap_score"]) ** weights["overlap"]
        )

        return df.sort_values("final_score", ascending=False).reset_index(drop=True)
