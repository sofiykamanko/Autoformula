# AutoFormula: library for feature engineering
### Спеціальні мови програмування

Манько Софія, Головач Лук'яна

**AutoFormula** is a Python library for automated feature analysis, evaluation and feature engineering in data analysis and ML tasks. The library is built to work with real-world, mixed-type datasets containing numerical, categorical, boolean, datetime and text features.

The **main goal** of **AutoFormula** is to provide a statistically grounded framework for answering important questions:
- Which features are useful for predicting the target?
- Which features duplicate the same information?
- Which features have a real statistical effect?
- Which feature transformations may be helpful?

The library works without machine learning models and relies on statistical methods.

**Main Functionality**

1. Data Preparation and Feature Types (remove duplicates, detect semantic feature types, find mismatches between data types, convert selected features to correct types).
2. Feature Profiling (compute basic statistics for each feature6 analyze correlations between numeric features, relationships between categorical features using Cramér’s V, visualize correlations).
3. Feature Analysis (measure separability between features and the target variable, estimate statistical impact of features, detect feature redundancy and overlap, automatically choose methods based on feature and target types).
4. Feature Quality Scoring (calculate a single quality score for each feature, combine informativeness, statistical impact and low redundancy for ranking features by their overall usefulness).
5. Feature Engineering (generate common non-linear transformations for numeric features, create feature interactions with a controlled number of combinations, support custom user-defined feature transformations).

**AutoFormula** can be used:
- for exploratory data analysis (EDA);
- for feature selection before training machine learning models;
- in educational and research projects;
- as a simple and transparent feature engineering tool.

  
