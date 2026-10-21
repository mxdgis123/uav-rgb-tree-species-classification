from pathlib import Path



import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score, precision_score, recall_score
from sklearn.model_selection import PredefinedSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data_loader import load_date, make_global_fold_map, scan_anchors
from features import FEATURE_NAMES
from shap_analysis import classwise_abs_shap_sum, save_class_stacked_plot


PROJECT_DIR = Path(__file__).resolve().parent
BASE_DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs"
IMAGE_SIZE = (90, 90)
K_FOLDS = 5
ANCHOR_SEED = 5
MODEL_SEED = 7
MODEL_NAME = "rf"
TIME_POINTS = [
    "20210510", "20210513", "20210517",
    "20210520", "20210524", "20210527",
    "20210602", "20210607",
    "20210623", "20210701", "20210708", "20210714",
    "20210720", "20210723",
    "20210726", "20210730", "20210806",
    "20210810", "20210812",
    "20210817", "20210820", "20210824", "20210827", "20210831",
    "20210904", "20210907", "20210916", "20210922",
    "20210924", "20210927", "20210930",
    "20211008", "20211012", "20211015", "20211021",
]
SHAP_TIME_POINTS = {"20210510"}
SHAP_TOP_N = 20
SHAP_X_MAX = None
MODEL_LABELS = {"rf": "RF", "et": "ET", "xgboost": "XGBoost"}


def make_model(n_classes):
    if MODEL_NAME == "rf":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_split=15,
            min_samples_leaf=6,
            ccp_alpha=0.01,
            class_weight="balanced",
            random_state=MODEL_SEED,
            n_jobs=-1,
        )

    if MODEL_NAME == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=100,
            max_depth=2,
            learning_rate=0.05,
            subsample=0.7,
            colsample_bytree=0.7,
            objective='multi:softmax',
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=MODEL_SEED,
            n_jobs=-1,
        )

    if MODEL_NAME == "et":
        return ExtraTreesClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_split=10,
            min_samples_leaf=6,
            ccp_alpha=0.01,
            class_weight="balanced",
            random_state=MODEL_SEED,
            n_jobs=-1,
        )

    raise ValueError(f"Unsupported model: {MODEL_NAME}")


def make_pipeline(n_classes):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", make_model(n_classes)),
    ])


def metrics(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Kappa": cohen_kappa_score(y_true, y_pred),
        "Precision (macro)": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall (macro)": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "Precision (weighted)": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "Recall (weighted)": recall_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def run_date(date, class_names, fold_map):
    date_dir = BASE_DATA_DIR / date
    if not date_dir.is_dir():
        print(f"Skipping missing date: {date_dir}")
        return None

    X, y, sample_ids = load_date(date_dir, class_names, IMAGE_SIZE)
    n_classes = len(class_names)
    if len(np.unique(y)) != n_classes:
        raise ValueError(f"Missing classes in {date}")

    folds = np.asarray([fold_map[sample_id] for sample_id in sample_ids])
    if set(folds) != set(range(K_FOLDS)):
        raise ValueError(f"Incomplete fold coverage in {date}")

    train_results = []
    predictions = np.full(len(y), -1)
    run_shap = date in SHAP_TIME_POINTS
    shap_totals = np.zeros((len(FEATURE_NAMES), n_classes)) if run_shap else None

    print(f"\n{date} | {MODEL_LABELS[MODEL_NAME]} | {len(y)} samples")
    for fold, (train_idx, valid_idx) in enumerate(PredefinedSplit(folds).split(), 1):
        if len(np.unique(y[train_idx])) != n_classes:
            raise ValueError(f"Missing training classes in {date}, fold {fold}")
        model = make_pipeline(n_classes)
        model.fit(X[train_idx], y[train_idx])
        train_results.append(metrics(y[train_idx], model.predict(X[train_idx])))
        predictions[valid_idx] = model.predict(X[valid_idx])
        if run_shap:
            shap_totals += classwise_abs_shap_sum(model, X[valid_idx], n_classes)
        print(f"  Fold {fold}/{K_FOLDS} complete")

    if np.any(predictions < 0):
        raise ValueError(f"Incomplete validation predictions: {date}")

    train_means = {
        key: np.mean([fold[key] for fold in train_results])
        for key in train_results[0]
    }
    valid_results = metrics(y, predictions)
    print(f"{'Metric':<24} {'Train':>10} {'Validation':>12}")
    for key in train_means:
        print(f"{key:<24} {train_means[key]:>10.4f} {valid_results[key]:>12.4f}")

    class_recalls = recall_score(
        y, predictions, labels=np.arange(n_classes), average=None, zero_division=0
    )
    print("\nValidation recall by species:")
    for name, score in zip(class_names, class_recalls):
        print(f"  {name:<30} {score:.4f}")

    final_model = make_pipeline(n_classes)
    final_model.fit(X, y)
    final_model.class_names_ = tuple(class_names)
    final_model.feature_names_ = tuple(FEATURE_NAMES)
    final_model.image_size_ = IMAGE_SIZE
    final_model.date_ = date
    model_dir = OUTPUT_DIR / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{date}_{MODEL_LABELS[MODEL_NAME]}.joblib"
    joblib.dump(final_model, model_path)
    print(f"Model saved: {model_path}")

    if run_shap:
        plot_path = save_class_stacked_plot(
            shap_totals / len(y), FEATURE_NAMES, class_names,
            date, MODEL_LABELS[MODEL_NAME], OUTPUT_DIR / "shap",
            top_n=SHAP_TOP_N, x_max=SHAP_X_MAX,
        )
        print(f"SHAP plot saved: {plot_path}")

    return train_means, valid_results, class_recalls


def main():
    if MODEL_NAME not in MODEL_LABELS:
        raise ValueError(f"Unsupported model: {MODEL_NAME}")

    sample_ids, class_names = scan_anchors(BASE_DATA_DIR, TIME_POINTS)
    fold_map = make_global_fold_map(sample_ids, K_FOLDS, ANCHOR_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result_file = OUTPUT_DIR / f"cv_metrics_{MODEL_LABELS[MODEL_NAME]}.txt"

    columns = [
        "Date", "Train_Accuracy", "Train_Kappa",
        "Validation_Accuracy", "Validation_Kappa",
    ] + [f"Recall_{name}" for name in class_names]

    with result_file.open("w", encoding="utf-8") as file:
        file.write("\t".join(columns) + "\n")
        for date in TIME_POINTS:
            result = run_date(date, class_names, fold_map)
            if result is None:
                continue

            train_means, valid_results, class_recalls = result
            values = [
                date,
                f"{train_means['Accuracy']:.6f}",
                f"{train_means['Kappa']:.6f}",
                f"{valid_results['Accuracy']:.6f}",
                f"{valid_results['Kappa']:.6f}",
            ] + [f"{recall:.6f}" for recall in class_recalls]
            file.write("\t".join(values) + "\n")
            file.flush()

    print(f"Results saved: {result_file}")


if __name__ == "__main__":
    main()
