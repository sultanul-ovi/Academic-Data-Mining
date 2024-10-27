import numpy as np
import re
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.feature_selection import SelectKBest, chi2, VarianceThreshold
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from imblearn.pipeline import make_pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.naive_bayes import BernoulliNB
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, StratifiedKFold, learning_curve
import random
import time

np.random.seed(42)
random.seed(42)

def load_data(file_path, has_labels=True):
    with open(file_path, "r") as fr:
        lines = fr.readlines()
    
    if has_labels:
        labels = [int(l[0]) for l in lines]
        docs = [re.sub(r'[^\w]', ' ', l[1:]).split() for l in lines]
    else:
        labels = None
        docs = [re.sub(r'[^\w]', ' ', l).split() for l in lines]
    
    features = []
    for doc in docs:
        line = [0] * 100001
        for value in doc:
            line[int(value)] = 1
        features.append(line)
    
    return np.array(features), np.array(labels) if labels is not None else None

print("Loading data...")
start_time = time.time()
X_train, y_train = load_data("train.dat", has_labels=True)
X_test, _ = load_data("test.dat", has_labels=False)
print(f"Data loaded in {time.time() - start_time:.2f} seconds.")

print("Applying VarianceThreshold...")
selector = VarianceThreshold(threshold=0.01)
X_train_selected = selector.fit_transform(X_train)
X_test_selected = selector.transform(X_test)

print("\nData Summary:")
print(f"Training Samples: {X_train_selected.shape[0]}, Features: {X_train_selected.shape[1]}")
print(f"Test Samples: {X_test_selected.shape[0]}")
train_dist = Counter(y_train)
print(f"Class Distribution - Class 0: {train_dist[0]}, Class 1: {train_dist[1]}")

classifier = BernoulliNB()
pipeline = make_pipeline(
    RandomUnderSampler(random_state=42),
    SelectKBest(chi2),
    clone(classifier)
)

param_grid = {'selectkbest__k': list(range(613, 614))}
grid_search = GridSearchCV(pipeline, param_grid, scoring='f1_weighted', cv=5, n_jobs=-1)

print("\nPerforming Grid Search...")
start_time = time.time()
grid_search.fit(X_train_selected, y_train)
print(f"Grid search completed in {time.time() - start_time:.2f} seconds.")

print("\n=== Cross-Validation Results ===")
cv_results = grid_search.cv_results_
for k, mean_score, std_score in zip(cv_results['param_selectkbest__k'], 
                                    cv_results['mean_test_score'], 
                                    cv_results['std_test_score']):
    print(f"k = {k}: F1 Score = {mean_score:.4f} (±{std_score:.4f})")

best_k = grid_search.best_params_['selectkbest__k']
best_f1 = grid_search.best_score_
print(f"\nBest k: {best_k}, Best F1 Score: {best_f1:.4f}")

print("\nTraining final model...")
final_pipeline = make_pipeline(
    RandomUnderSampler(random_state=42),
    SelectKBest(chi2, k=best_k),
    clone(classifier)
)
final_pipeline.fit(X_train_selected, y_train)

y_train_pred = final_pipeline.predict(X_train_selected)
print("\n=== Training Performance ===")
print(classification_report(y_train, y_train_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_train, y_train_pred))

print("\nPredicting on test data...")
test_predictions = final_pipeline.predict(X_test_selected)

with open('result_613.dat', 'w') as output:
    for pred in test_predictions:
        output.write(f"{pred}\n")
print("Predictions saved to result_613.dat.")

def process_dat_file(file_path):
    count_0 = 0
    count_1 = 0
    line_numbers_of_1s = []

    with open(file_path, 'r') as file:
        for line_num, line in enumerate(file, start=1):
            value = line.strip()
            if value == '0':
                count_0 += 1
            elif value == '1':
                count_1 += 1
                line_numbers_of_1s.append(line_num)

    return count_0, count_1, line_numbers_of_1s


file_path = 'result_613.dat'
count_0, count_1, line_numbers_of_1s = process_dat_file(file_path)
print(f"Number of 0s: {count_0}")
print(f"Number of 1s: {count_1}")
print(f"Line numbers of 1s: {line_numbers_of_1s}")
