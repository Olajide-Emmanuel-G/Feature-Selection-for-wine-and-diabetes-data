# Feature-Selection-for-wine-and-diabetes-data

A pair of Jupyter notebooks and written reports that explore **feature selection**, the process of deciding which columns of a dataset a machine learning model should actually use. The project works through the main families of feature selection on two classic datasets and documents what each technique does, how it behaves, and what can be learned from the results.

**Author:** Olajide Emmanuel

## Project Overview
This project looks at feature selection from two angles:

- **Version 1 (Wine):** tries several selection methods on the same problem and compares them by measuring how well a model performs with each chosen feature set.
- **Version 2 (Diabetes):** organises the methods by their three main families (filter, wrapper and embedded) and shows one example of each, focusing on how each one works and what it produces.

Each notebook is paired with a detailed written report that explains every concept, the reasoning behind each step, and the main observations.

---

## Datasets

### Wine dataset (Version 1)

| | |
|---|---|
| **Source** | Built into scikit-learn (`sklearn.datasets.load_wine`) |
| **Samples** | 178 |
| **Features** | 13 chemical measurements (for example alcohol, malic acid, flavanoids, colour intensity, proline) |
| **Target** | 3 classes, representing three grape cultivars grown in the same region of Italy |
| **Task** | Multi-class classification |

The dataset is small, fully numeric and free of missing values, which makes it a clean setting for comparing feature selection methods without other data problems getting in the way.

### Pima Indians diabetes dataset (Version 2)

| | |
|---|---|
| **Source** | Loaded from a public CSV link with `pandas.read_csv` |
| **Samples** | 768 |
| **Features** | 8 medical measurements: `preg`, `plas`, `pres`, `skin`, `test`, `mass`, `pedl`, `age` |
| **Target** | `class`: 0 = no diabetes, 1 = diabetes |
| **Task** | Binary classification |

| Column | Meaning |
|---|---|
| `preg` | Number of times pregnant |
| `plas` | Plasma glucose concentration |
| `pres` | Diastolic blood pressure |
| `skin` | Triceps skinfold thickness |
| `test` | 2-hour serum insulin |
| `mass` | Body mass index (BMI) |
| `pedl` | Diabetes pedigree function (a score for family history) |
| `age` | Age in years |

The data comes without a header row, so column names are supplied when loading it. The column descriptions above follow the dataset's standard documentation.

---

## What Each Version Covers

### Version 1: Wine Data Feature Selection

Version 1 is a comparison study. It trains a **Gradient Boosting Classifier** on all 13 features to set a baseline, then applies four feature selection methods and measures each one with the same model and the same metric (weighted F1 score) on the same train/test split.

The notebook walks through:

1. Loading the wine data into a pandas DataFrame
2. Visual exploration with a swarm plot, and a class balance check
3. A stratified 70/30 train/test split
4. A baseline model using all features
5. Four feature selection methods, each tested against the baseline
6. A final side-by-side comparison chart

**Results (weighted F1 on the test set):**

| Approach | Features kept | Weighted F1 |
|---|---|---|
| All features (baseline) | 13 | 0.908 |
| Variance threshold | 11 | 0.963 |
| Mutual information (SelectKBest) | 3 | 0.981 |
| Recursive Feature Elimination (RFE) | 3 | 1.000 |
| Boruta | 9 | 0.982 |

Features chosen with three features kept:

- **Mutual information:** flavanoids, color_intensity, proline
- **RFE:** color_intensity, od280/od315_of_diluted_wines, proline

### Version 2: Diabetes Data Feature Selection

Version 2 is a conceptual tour. It applies one representative method from each family of feature selection to the diabetes data and examines what each one reports. It does not train a final model or compute a score, so its purpose is to understand how each family works.

The notebook walks through:

1. Loading data from a URL with custom column names
2. Quick checks with `head()` and `shape`
3. Separating features (`x`) and target (`y`) as NumPy arrays
4. A **filter** method: chi-square with `SelectKBest`
5. A **wrapper** method: RFE with Logistic Regression
6. An **embedded** method: Ridge regression (L2 regularization) with a helper function to print coefficients

**What each method selected:**

| Method | Family | Result |
|---|---|---|
| Chi-square, `SelectKBest` (k=4) | Filter | plas, test, mass, age |
| RFE with Logistic Regression (3 features) | Wrapper | preg, mass, pedl |
| Ridge regression (alpha=1.0) | Embedded | Largest weights: pedl, preg, mass |

---

## Topics and Techniques Explored

**Feature selection methods**

| Technique | Family | Used in |
|---|---|---|
| Variance threshold (with MinMax scaling) | Filter | Version 1 |
| `SelectKBest` with mutual information | Filter | Version 1 |
| `SelectKBest` with chi-square | Filter | Version 2 |
| Recursive Feature Elimination (RFE) | Wrapper | Versions 1 and 2 |
| Boruta (shadow-feature comparison) | Wrapper | Version 1 |
| Ridge regression / L2 regularization | Embedded | Version 2 |

**Modelling and evaluation**

- Gradient Boosting Classifier and Logistic Regression
- Stratified train/test splitting with a fixed random seed
- Weighted F1 score for multi-class evaluation
- Baseline-versus-selected-features comparison

**Data handling and visualisation**

- Building DataFrames from scikit-learn datasets and loading CSV data from a URL
- Reshaping data with `pd.melt` for plotting
- Swarm plots and annotated bar charts with matplotlib and seaborn
- Working on copies of the data so that each method starts from the same untouched inputs
- Recovering feature names from selector masks (`get_support`, `support_`, `ranking_`)

---

## Repository Structure

```
.
├── README.md
├── Wine Data (Version 1).ipynb
├── Diabetes Data (Version 2).ipynb
└── Documentation.docx

```

## Key Takeaways

**From Version 1 (Wine)**

- **Fewer features can beat all of them.** Every selection method scored higher than the 13-feature baseline of 0.908, and the best results came from keeping only three features. The extra columns were adding noise more than information.
- **Performance rises, then levels off.** In both the mutual information and RFE results, scores climb quickly up to about three features and then flatten or decline. The first few strong features do most of the work.
- **Independent methods agree on the strongest features.** Proline and colour intensity were chosen by mutual information, RFE and Boruta alike, which is good evidence that they matter.
- **Low variance does not mean low value.** The variance threshold removed ash and magnesium, but Boruta kept both. Variance measures spread, not usefulness for predicting the class.
- **Scaling is needed to compare variance fairly.** Raw variances differed enormously because features use different units, so MinMax scaling was needed before choosing a threshold.

**From Version 2 (Diabetes)**

- **The three families give different answers.** Filter, wrapper and embedded methods ask different questions of the data, so they selected different features. Only `mass` (BMI) appeared in all three.
- **Treat methods as complementary opinions.** Agreement between methods builds confidence in a feature, and disagreement is a reason to look more closely.
- **Feature scale affects results.** Chi-square scores and Ridge coefficients depend on the units of each feature, so scaling should be considered before reading them. In an additional check, standardising the features changed which features RFE selected and which Ridge weight was largest.
- **Data quality still matters.** Columns such as `skin`, `test`, `plas`, `pres` and `mass` contain zeros that cannot be real measurements and almost certainly stand for missing values. They can quietly influence every method.

**Across both versions**

- Feature selection is part of building a good model and not an optional extra.
- A selection result is only a claim until it is tested by training a model on the chosen features and scoring it on held-out data.
- A reliable workflow is to set a baseline, try methods from different families, keep everything else fixed, and compare the results.

