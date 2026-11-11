# Learning to Rank for Information Retrieval using the Cranfield Dataset

An end-to-end Machine Learning-based Information Retrieval system that learns to rank documents according to their relevance to a given query.

This project implements and compares three Learning-to-Rank (LTR) algorithms:

- **Pairwise Linear SVM**
- **XGBoost Rank:Pairwise**
- **LightGBM LambdaMART**

The models are trained and evaluated using the Cranfield Information Retrieval Test Collection. After comparing their performance, the trained **LightGBM LambdaMART model** is selected for the final search application.

The application supports free-text queries and retrieves the top-K most relevant documents from the Cranfield collection.

---

## 1. Project Overview

Traditional Information Retrieval (IR) methods, such as TF-IDF and BM25, rely on predefined scoring functions to rank documents. Learning-to-Rank approaches use machine learning to learn ranking patterns from query-document features and relevance judgments.

This project implements a complete Learning-to-Rank pipeline:

1. Preprocess the Cranfield documents and queries.
2. Extract numerical features from query-document pairs.
3. Construct a labeled dataset using relevance judgments.
4. Train three Learning-to-Rank models.
5. Tune model hyperparameters using cross-validation.
6. Evaluate and compare model performance.
7. Select the best-performing model based on ranking performance.
8. Integrate the selected model into a free-text document search application.

### Models Implemented

| Model | Ranking Approach |
|---|---|
| Pairwise Linear SVM | Pairwise learning |
| XGBoost Rank:Pairwise | Gradient-boosted pairwise ranking |
| LightGBM LambdaMART | LambdaRank-based ranking |

---

## 2. Dataset

This project uses the **Cranfield Information Retrieval Test Collection**, a benchmark dataset commonly used for evaluating information retrieval systems.

| Dataset Component | Count |
|---|---:|
| Documents | 1,400 |
| Queries | 225 |
| Relevance judgments | As provided in the qrels file |

The collection contains:

- A document collection containing the text to be retrieved.
- A query collection containing information needs expressed as queries.
- Relevance judgments (qrels) describing the relevance of documents to queries.

### Relevance Labels

The original Cranfield judgments use codes 1 to 4, where **1 is the most relevant** (a complete answer) and 4 is the least (minimum interest). Because ranking metrics and the models treat larger values as more relevant, the labels are remapped in `build_dataset.py`:

| Original code | Meaning | Label used |
|---:|---|---:|
| 1 | Complete answer | 4 |
| 2 | High degree of relevance | 3 |
| 3 | Useful background | 2 |
| 4 | Minimum interest | 1 |
| -1 / unjudged | Not relevant | 0 |

**Unjudged documents:** query-document pairs without a judgment are assigned label `0`. This is an experimental assumption that treats unjudged documents as non-relevant, and it may affect both training and evaluation.

---

## 3. Installation and Setup

### 3.1 Clone the Repository

```bash
git clone <repository-url>
cd <repository-directory>
```

### 3.2 Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment:

**Windows**

```bash
venv\Scripts\activate
```

**Linux/macOS**

```bash
source venv/bin/activate
```

### 3.3 Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Data Preprocessing

The preprocessing stage prepares the raw Cranfield documents and queries for feature extraction.

It applies consistent text-processing operations, including:

- Unicode normalization using NFKC.
- Tokenization.
- Stopword removal.
- Porter stemming.

The same preprocessing pipeline is used for documents, training queries, and free-text queries submitted to the search application.

Run:

```bash
python preprocessing.py
```

### Expected Output

```text
data/
└── preprocessed/
    ├── documents_processed.csv
    └── queries_processed.csv
```

The processed document file contains the tokenized and stemmed document fields, while the processed query file contains the corresponding query tokens.

---

## 5. Feature Extraction and Dataset Construction

### 5.1 Feature Extraction

The system extracts **110 numerical features** for each query-document pair.

Features are calculated across five document representations:

1. Body
2. Author
3. Title
4. Bibliography
5. Whole document

Each representation contributes 22 features.

### Feature Categories

The extracted features describe the relationship between a query and a document. They include:

- Term frequency statistics.
- Logarithmic term frequency.
- Inverse document frequency (IDF).
- TF-IDF-related statistics.
- Query-term occurrence and coverage statistics.
- Query-document cosine similarity.
- BM25-style relevance scores.
- Boolean OR and AND matching features.
- Document and query statistics.

### 5.2 Building the Training Dataset

Run:

```bash
python build_dataset.py
```

This stage:

1. Loads the preprocessed documents and queries.
2. Generates feature vectors for every query-document pair.
3. Reads the Cranfield relevance judgments.
4. Assigns relevance labels to judged pairs.
5. Assigns relevance `0` to unjudged pairs, according to the project's assumption.
6. Combines the features and relevance labels into a training dataset.

**Output:**

```text
data/
└── features/
    └── dataset.csv
```

### Dataset Format

Each row represents one query-document pair.

| Column | Description |
|---|---|
| `qid` | Query ID |
| `docid` | Document ID |
| Feature columns | 110 numerical query-document features |
| `relevance` | Ground-truth relevance label |

This dataset is shared by the three Learning-to-Rank models.

---

## 6. Learning-to-Rank Models

Three algorithms are implemented to investigate different approaches to learning document relevance and ranking.

### 6.1 Pairwise Linear SVM

The Linear SVM model uses a pairwise ranking approach.

Rather than independently predicting the relevance of each document, it learns to distinguish between pairs of documents with different relevance labels.

For a pair in which document A is more relevant than document B, the model learns from the difference between their feature vectors.

**Implementation details:**

- Pairwise training examples.
- Feature standardization using `StandardScaler`.
- `LinearSVC` with squared hinge loss.
- Query-level train/test split.
- `GroupKFold` cross-validation.
- Hyperparameter tuning using the regularization parameter `C`.

At inference time, the model uses its decision function to generate document ranking scores.

### 6.2 XGBoost Rank:Pairwise

XGBoost is implemented using its built-in pairwise ranking objective:

```python
objective="rank:pairwise"
```

Unlike the Linear SVM implementation, XGBoost learns pairwise ranking relationships internally.

**Implementation details:**

- `XGBRanker`.
- Pairwise ranking objective.
- Query-grouped training data.
- `GroupKFold` cross-validation.
- Optuna-based hyperparameter optimization.
- GPU acceleration where supported.

The model learns non-linear relationships between query-document features.

### 6.3 LightGBM LambdaMART

LightGBM is implemented using the LambdaMART ranking objective:

```python
objective="lambdarank"
```

LambdaMART is a gradient-boosted decision-tree ranking algorithm that uses LambdaRank-style gradients to optimize ranking quality.

**Implementation details:**

- `LGBMRanker`.
- LambdaMART objective.
- Query-grouped training data.
- `GroupKFold` cross-validation.
- Optuna-based hyperparameter optimization.
- NDCG@10 as the hyperparameter optimization objective.

The trained LambdaMART model is used by the final search application.

---

## 7. Model Training and Hyperparameter Optimization

Each Learning-to-Rank model is implemented in a separate Python script. Models can be trained independently, allowing their training processes, hyperparameter optimization, and results to be generated separately.

### 7.1 Train/Test Split

All three models use the same query-level train/test splitting strategy:

| Split | Proportion |
|---|---:|
| Training queries | 80% |
| Testing queries | 20% |

The split uses:

```python
random_state=42
```

Splitting is performed by query ID rather than by individual query-document rows. This prevents documents associated with the same query from appearing in both the training and testing sets.

### 7.2 Cross-Validation

The training set uses five-fold GroupKFold cross-validation:

```python
GroupKFold(n_splits=5)
```

Grouping by query ID ensures that all document pairs belonging to a query remain within the same fold.

### 7.3 Hyperparameter Optimization

Optuna is used to optimize the XGBoost and LightGBM models.

The optimization process uses:

- TPE (Tree-structured Parzen Estimator) sampling.
- Five-fold GroupKFold cross-validation.
- Mean validation NDCG@10 as the optimization objective.

The best hyperparameters are selected based on validation performance.

### 7.4 Training the Models

Navigate to the `experiments` directory from the project root:

```bash
cd experiments
```

Run each model independently:

**Train Pairwise Linear SVM:**

```bash
python Pairwise_Linear_Ranking_SVM.py
```

**Train XGBoost Rank:Pairwise:**

```bash
python Pairwise_XGBoost_Rank.py
```

**Train LightGBM LambdaMART:**

```bash
python lambdamart.py
```

Each script performs the corresponding model training and saves the trained model and relevant training results.

After running all three scripts, the trained models and evaluation outputs will be available for comparison.

---

## 8. Model Evaluation and Comparison

After running all three scripts, return to the project root directory:

```bash
cd ..
```

Then proceed to model evaluation and comparison using `compare_models.py`.

```bash
python compare_models.py
```

The script:

1. Loads the three trained models.
2. Rebuilds the same held-out test queries (query-level 80/20 split, `random_state=42`).
3. Scores the test documents with each model **and with a BM25 baseline**.
4. Computes the ranking metrics for every ranker.
5. Runs paired Wilcoxon signed-rank tests on per-query results.

### 8.1 BM25 Baseline

The baseline ranks documents by the whole-document BM25 feature alone (`whole_document_bm25`), with no learning. It uses the same test queries and the same evaluation code as the learned models, so it shows how much learning adds over a strong lexical retrieval function.

### 8.2 Evaluation Metrics

| Metric | Description |
|---|---|
| MAP | Mean Average Precision |
| MRR | Mean Reciprocal Rank |
| P@5, P@10 | Precision at 5 and 10 |
| Recall@5, Recall@10 | Recall at 5 and 10 |
| nDCG@5, nDCG@10 | Normalized Discounted Cumulative Gain |
| Pairwise Accuracy | Proportion of document pairs with different labels that are ordered correctly |

For MAP, MRR, Precision and Recall, relevance is binary (`relevance > 0`). For nDCG, the graded labels (0 to 4) are used.

**Tie-breaking:** when documents receive equal scores, the less relevant document is ranked first (pessimistic tie-breaking). This matters on Cranfield, where many documents share no terms with a query and score identically (about 36% of test rows have a BM25 score of 0). Ties therefore never inflate the results.

### 8.3 Results on the Held-Out Test Queries (45 queries, 63,000 query-document pairs)

| Rank | Ranker | MAP | MRR | P@5 | P@10 | Recall@5 | Recall@10 | nDCG@5 | nDCG@10 | Pairwise Acc. |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | LightGBM LambdaMART | 0.3784 | 0.6568 | **0.4222** | **0.2911** | 0.3693 | **0.4952** | 0.4396 | **0.4695** | 0.9146 |
| 2 | XGBoost Rank:Pairwise | **0.3821** | **0.6849** | **0.4222** | 0.2756 | **0.3934** | 0.4734 | **0.4551** | 0.4628 | **0.9227** |
| 3 | BM25 (baseline) | 0.3776 | 0.6297 | 0.3778 | 0.2756 | 0.3792 | 0.4669 | 0.4262 | 0.4560 | 0.9161 |
| 4 | Linear SVM | 0.3772 | 0.6712 | 0.3822 | 0.2733 | 0.3544 | 0.4707 | 0.4129 | 0.4479 | 0.9197 |

*Rows are ordered by nDCG@10. The best value in each column is in bold.*

### 8.4 Statistical Significance (paired Wilcoxon signed-rank test)

Per-query nDCG@10 and MAP are compared with a paired Wilcoxon signed-rank test over the 45 test queries (`significance_tests.csv`).

**nDCG@10**

| Comparison (A vs B) | Mean difference (A - B) | Queries A better / B better / tied | p-value | Significant at 0.05 |
|---|---:|:---:|---:|:---:|
| LightGBM LambdaMART vs BM25 | +0.0135 | 21 / 16 / 8 | 0.455 | No |
| XGBoost Rank:Pairwise vs BM25 | +0.0068 | 20 / 17 / 8 | 0.603 | No |
| Linear SVM vs BM25 | -0.0081 | 19 / 16 / 10 | 0.974 | No |
| LightGBM LambdaMART vs XGBoost Rank:Pairwise | +0.0067 | 19 / 18 / 8 | 0.712 | No |
| LightGBM LambdaMART vs Linear SVM | +0.0216 | 21 / 13 / 11 | 0.153 | No |
| XGBoost Rank:Pairwise vs Linear SVM | +0.0149 | 22 / 12 / 11 | 0.074 | No |

**MAP**

| Comparison (A vs B) | Mean difference (A - B) | Queries A better / B better / tied | p-value | Significant at 0.05 |
|---|---:|:---:|---:|:---:|
| LightGBM LambdaMART vs BM25 | +0.0007 | 25 / 18 / 2 | 0.507 | No |
| XGBoost Rank:Pairwise vs BM25 | +0.0045 | 27 / 17 / 1 | 0.318 | No |
| Linear SVM vs BM25 | -0.0004 | 22 / 20 / 3 | 0.703 | No |
| LightGBM LambdaMART vs XGBoost Rank:Pairwise | +0.0037 | 18 / 24 / 3 | 0.544 | No |
| LightGBM LambdaMART vs Linear SVM | +0.0012 | 19 / 24 / 2 | 0.837 | No |
| XGBoost Rank:Pairwise vs Linear SVM | +0.0049 | 25 / 18 / 2 | 0.378 | No |

*p-values are not corrected for multiple comparisons. No difference is significant even without correction.*

### 8.5 Result Analysis

- **No learned model significantly outperforms BM25.** LightGBM LambdaMART has the highest nDCG@10 (0.4695 vs 0.4560 for BM25), but the gain of 0.0135 is not statistically significant (p = 0.455). The same holds for XGBoost and for MAP, where all four rankers are within 0.005 of each other.
- **The learned models differ little from one another.** The largest nDCG@10 gap is between XGBoost and the Linear SVM (+0.0149, p = 0.074), which comes closest to significance but does not reach it.
- **Where the tree models show an edge.** Both boosted models have a higher P@5 than BM25 (0.4222 vs 0.3778) and a higher nDCG@5 (0.4396 and 0.4551 vs 0.4262). This is the largest observed gap, but it was not tested for significance, so it should be treated as a possible trend only.
- **LightGBM LambdaMART** leads on P@10, Recall@10 and nDCG@10, which are the top-10 metrics.
- **XGBoost Rank:Pairwise** leads on MAP, MRR, Recall@5, nDCG@5 and pairwise accuracy.
- **Linear SVM** performs about the same as BM25, with no consistent advantage. It scores above BM25 on some metrics (MRR, P@5, Recall@10) and below on others (MAP, nDCG@5, nDCG@10).
- **Pairwise accuracy is not informative here.** All rankers score about 0.91 to 0.92, including BM25, because most of the counted pairs are relevant-vs-irrelevant pairs that are easy to order.
- **Why the gains are small.** Cranfield is small, BM25 is a strong baseline on it, and many of the 110 features are themselves BM25 or TF-IDF variants, so the learned models start with little extra signal beyond what BM25 already captures.

### 8.6 Final Model Selection

LightGBM LambdaMART is used in the search application because it has the highest test nDCG@10 among the learned models and the best top-10 precision and recall. The margin over XGBoost (p = 0.712) and over BM25 (p = 0.455) is not statistically significant, so the choice is a practical one rather than evidence that LambdaMART is better. Selecting on the test set is also a limitation. The cross-validation scores stored in each model's metadata (`best_cv_ndcg_at_10` and the equivalents) are the more appropriate basis for selection.

### 8.7 Generated Outputs

```text
model/model_comparison/
├── model_comparison.csv       # mean metrics per ranker, including BM25
├── per_query_metrics.csv      # per-query metrics for every ranker
├── significance_tests.csv     # paired Wilcoxon results
└── ranked_test_documents.csv  # ranked test documents (large; not committed)
```

### 8.8 Limitations

- **Small test set:** one query-level split with 45 test queries. A different split could change the ranking of the models. Repeated splits or nested cross-validation would give more reliable estimates.
- **Unjudged = non-relevant:** unjudged documents are labeled 0, which may penalize rankers that retrieve relevant but unjudged documents.
- **Baseline definition:** the baseline uses the project's own whole-document BM25 feature, not an independently tuned BM25 implementation.
- **Multiple comparisons:** the reported p-values are uncorrected, and only nDCG@10 and MAP were tested.
- **Model selection on the test set:** the final model was chosen using test nDCG@10, although the differences are within noise.

## 9. Running the Search Application

After comparing the three models, the selected **LightGBM LambdaMART model** is used in the final search application.

The application is implemented in `main.py` and loads the saved model from `model / lambdamart / lambdamart_model.joblib`

### 9.1 Interactive Search

Run:

```bash
python main.py
```

Enter a free-text query when prompted. The application retrieves the top 10 documents by default.

### 9.2 Single Free-Text Query

```bash
python main.py --query "Aircraft structural design"
```

### 9.3 Batch Query Search

Create a text file containing one query per line:

```text
Aircraft structural design
Aerodynamic characteristics of wings
Jet engine performance
```

Run:

```bash
python main.py --query-file queries.txt
```

### 9.4 Retrieve a Custom Number of Results

For example, retrieve the top 20 documents:

```bash
python main.py --query "Aircraft structural design" --top-k 20
```

### 9.5 Specify an Output File

```bash
python main.py --query "Aircraft structural design" --output results.txt
```

### How Search Works

For each submitted query, the application:

1. Accepts a free-text query from the user.
2. Applies the same preprocessing pipeline used during training.
3. Loads the preprocessed Cranfield document collection.
4. Extracts the 110 query-document features for every document.
5. Uses the trained LambdaMART model to predict a ranking score for each document.
6. Sorts documents by their predicted scores in descending order.
7. Returns the requested top-K document IDs.

The application ranks the full collection of 1,400 documents without using a separate candidate-retrieval stage.

**Note:** Cranfield query IDs are used during model training and evaluation. However, the final search application accepts arbitrary free-text queries, so users do not need to provide an existing Cranfield query ID.

---

## 10. Important Implementation Considerations

- **Query-level splitting:** Training and testing are separated by query ID to prevent query leakage.
- **Query grouping:** Documents belonging to the same query are kept together during ranking-model training and cross-validation.
- **Consistent feature extraction:** The feature definitions, feature order, and extraction configuration must remain identical between training and inference.
- **Consistent preprocessing:** Documents and user queries must undergo the same text-processing operations.
- **Relevance ordering:** Original graded relevance labels are preserved for training and nDCG evaluation.
- **Unjudged documents:** Unjudged query-document pairs are assigned relevance `0`, which is an experimental assumption.
- **Ranking inference:** The LambdaMART model predicts a score for each query-document pair. Documents are sorted by these scores to produce the final ranking.
- **Full collection ranking:** The current search application scores all 1,400 documents rather than retrieving candidates from a separate first-stage retrieval system.

---

## 11. Future Improvements

Several extensions could further improve the system:

- Perform feature ablation studies to understand the contribution of individual feature groups.
- Explore alternative feature engineering techniques.
- Investigate more sophisticated approaches to handling unjudged documents.
- Introduce a first-stage retrieval component for larger document collections.
- Add a cross-encoder model for neural reranking.
- Compare additional Learning-to-Rank algorithms.
- Develop a web-based search interface.
- Extend the system to larger and more diverse information retrieval datasets.

---

## 12. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core implementation |
| Pandas and NumPy | Data processing and numerical computation |
| NLTK | Text preprocessing and stemming |
| Scikit-learn | Linear SVM, preprocessing, and evaluation |
| XGBoost | Pairwise Learning-to-Rank |
| LightGBM | LambdaMART ranking |
| Optuna | Hyperparameter optimization |
| Joblib | Model serialization |
| SciPy | Paired Wilcoxon significance tests

---

## 13. Conclusion

This project presents an end-to-end Learning-to-Rank framework for information retrieval using the Cranfield dataset.

By implementing Pairwise Linear SVM, XGBoost Rank:Pairwise, and LightGBM LambdaMART, the project explores different machine learning approaches to learning relevance relationships between queries and documents.

The three models are trained and evaluated using query-document features and relevance judgments. Their performance is compared using standard information retrieval metrics, including MAP, MRR, Precision, Recall, and nDCG.

Based on the experimental results, LightGBM LambdaMART is selected for the final search application because of its stronger nDCG@10, P@10, and Recall@10 performance.

The final application integrates the trained LambdaMART model into a free-text search interface, allowing users to submit queries and retrieve the top-ranked documents from the Cranfield collection.

Overall, this project demonstrates how traditional information retrieval features and machine learning-based ranking algorithms can be combined to build a complete document retrieval and ranking system.
