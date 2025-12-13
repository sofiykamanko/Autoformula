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
        Compute Pearson, Spearman, and Kendall correlation matrices for numeric features.

        Parameters
        ----------
        df_num : pandas.DataFrame
            Input DataFrame. Only numeric and boolean columns are used.

        Returns
        -------
        dict[str, pandas.DataFrame]
            Correlation matrices: {"pearson", "spearman", "kendall"}.
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
        '''
        Plot correlation heatmaps from the dict returned by check_correlations().

        Parameters
        ----------
        corrs_dict : dict
            Dictionary with correlation matrices:
            {
                "pearson": pd.DataFrame,
                "spearman": pd.DataFrame,
                "kendall": pd.DataFrame
            }

        figsize : tuple, optional
            Size of the figure (default=(30, 10)).

        annot_size : int, optional
            Font size for correlation values (default=12).
        
        Returns
        -------
        None
            Displays correlation heatmaps using matplotlib/seaborn.
        '''

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
        Compute a pairwise Cramér’s V matrix for categorical and boolean columns.

        Parameters
        ----------
        df : DataFrame
            Input dataset; only categorical and boolean columns are used.
        show_plot : bool, default=True
            Whether to display a heatmap.

        Returns
        -------
        DataFrame
            Symmetric matrix of Cramérs V values.
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
        """
        Compute feature overlap scores.

        Overlap definition:
        - numeric + numeric - |Spearman| (Pearson / Kendall optional)
        - categorical + categorical - Cramér’s V
        - numeric + categorical - Correlation Ratio (η²)

        How to interpret results:
        - overlap_score ∈ [0, 1]
        - lower overlap_score - feature is more unique (better)
        - higher overlap_score - feature is redundant (duplicates other features)
        - overlap_score is computed as mean of top-k strongest overlaps per feature

        Parameters
        ----------
        df : pd.DataFrame
            Dataset with features only (no target).
        top_k : int, default=3
            Number of strongest overlaps used to aggregate feature score.
        use_pearson : bool, default=False
            Whether to include Pearson correlation for numeric-numeric pairs.
        use_kendall : bool, default=False
            Whether to include Kendall tau for numeric-numeric pairs.
        return_pairwise : bool, default=True
            Whether to return pairwise overlap table.

        Returns
        -------
        dict
            {
            "feature_overlap": pd.DataFrame,
            "pairwise_overlap": pd.DataFrame | None
            }
        """

        #helper functions

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
                    overlap = abs(spearmanr(x, y, nan_policy="omit")[0])
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
        """
        Compute feature–target separability scores with an optional permutation test.

        For each feature in `X`, the function selects an appropriate association
        measure based on feature/target types:
        - numeric–numeric: absolute Spearman correlation
        - categorical–numeric: correlation ratio (η²)
        - categorical–categorical: Cramér’s V

        Optionally performs a permutation test to estimate the significance of
        each separability score.

        Parameters
        ----------
        X : pandas.DataFrame
            Feature matrix.
        y : pandas.Series
            Target variable.
        run_permutation : bool, optional
            Whether to compute permutation-based p-values.
        n_permutations : int, optional
            Number of permutations used to estimate p-values.

        Returns
        -------
        pandas.DataFrame
            Ranked table with columns:
            - feature
            - separability_score
            - perm_p_value (if `run_permutation=True`)
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
        Compute statistical impact (effect size) of each feature on the target variable.

        For every feature in `X`, the function automatically selects an appropriate
        effect size metric and statistical test based on feature and target types:

        - numeric–numeric: |Spearman correlation|
        - numeric–categorical or categorical–numeric:
        η² (correlation ratio) + Kruskal–Wallis test
        - categorical–categorical:
        Cramér’s V + chi-square test

        Parameters
        ----------
        X : pandas.DataFrame
            Feature matrix.
        y : pandas.Series
            Target variable.

        Returns
        -------
        pandas.DataFrame
            Table sorted by decreasing impact with columns:
            - feature : feature name
            - impact_score : effect size (strength of association)
            - impact_metric : metric used to compute the effect size
            - test : statistical test applied
            - p_value : p-value of the corresponding test
        """

        results = []

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

                mean_total = values.mean()
                ss_between = 0.0

                for cat in categories.categories:
                    group = values[y == cat]
                    ss_between += len(group) * (group.mean() - mean_total) ** 2

                ss_total = ((values - mean_total) ** 2).sum()
                impact = ss_between / ss_total if ss_total > 0 else 0.0

                groups = [x[y == cat] for cat in categories.categories]
                _, p = kruskal(*groups)

                metric = "eta_squared"
                test = "kruskal"

            # categorical + numeric 
            elif not x_is_num and y_is_num:
                categories = pd.Categorical(x)
                values = y.astype(float)

                mean_total = values.mean()
                ss_between = 0.0

                for cat in categories.categories:
                    group = values[x == cat]
                    ss_between += len(group) * (group.mean() - mean_total) ** 2

                ss_total = ((values - mean_total) ** 2).sum()
                impact = ss_between / ss_total if ss_total > 0 else 0.0

                groups = [y[x == cat] for cat in categories.categories]
                _, p = kruskal(*groups)

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
        """
        Compute a composite feature quality score based on separability, impact, and overlap.

        The final score prioritizes features that:
        - are strongly associated with the target (separability),
        - have a large statistical effect size (impact),
        - provide non-redundant information (low overlap with other features).

        Parameters
        ----------
        X : pandas.DataFrame
            Feature matrix.
        y : pandas.Series
            Target variable.
        weights : dict, optional
            Exponents controlling the contribution of each component:
            {"separability", "impact", "overlap"}.
        top_k_overlap : int, optional
            Number of strongest overlaps used to compute mean feature redundancy.

        Returns
        -------
        pandas.DataFrame
            Features ranked by decreasing quality with columns:
            - separability_score
            - impact_score
            - overlap_score
            - final_score
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
