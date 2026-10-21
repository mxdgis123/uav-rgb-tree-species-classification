
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
import numpy as np


CLASS_COLORS = ("#b6c2a0", "#c7e3e6", "#b0d4d0", "#6eb0a9", "#efd4cc")


def classwise_abs_shap_sum(pipeline, X_valid, n_classes):
    import shap

    estimator = pipeline.named_steps["classifier"]
    X_scaled = pipeline.named_steps["scaler"].transform(X_valid)
    if len(estimator.classes_) != n_classes:
        raise ValueError("Missing classes in the training fold")

    explainer = shap.TreeExplainer(estimator)
    try:
        raw = explainer.shap_values(X_scaled, check_additivity=False)
    except TypeError:
        raw = explainer.shap_values(X_scaled)

    values = (np.stack(raw, axis=-1) if isinstance(raw, list)
              else np.asarray(getattr(raw, "values", raw)))
    n_samples, n_features = X_scaled.shape
    if values.shape == (n_classes, n_samples, n_features):
        values = np.moveaxis(values, 0, -1)
    elif values.shape == (n_samples, n_classes, n_features):
        values = np.swapaxes(values, 1, 2)
    elif values.shape == (n_samples, n_features) and n_classes == 2:
        values = np.stack([-values, values], axis=-1)

    if values.shape != (n_samples, n_features, n_classes):
        raise ValueError(f"Unexpected SHAP shape: {values.shape}")
    return np.abs(values).sum(axis=0)


def save_class_stacked_plot(values, feature_names, class_names, date,
                            model_name, output_dir, top_n=20, x_max=None):
    if values.shape != (len(feature_names), len(class_names)):
        raise ValueError(f"Unexpected class importance shape: {values.shape}")

    indices = np.argsort(values.sum(axis=1))[-top_n:]
    positions = np.arange(len(indices))
    offsets = np.zeros(len(indices))
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

    for i, class_name in enumerate(class_names):
        contributions = values[indices, i]
        color = CLASS_COLORS[i] if i < len(CLASS_COLORS) else plt.get_cmap("tab20")(i % 20)
        ax.barh(positions, contributions, left=offsets, height=0.8,
                color=color, label=class_name)
        offsets += contributions

    ax.set_yticks(positions)
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel("Mean absolute SHAP value", fontsize=12)
    ax.set_ylabel("Feature", fontsize=12)
    ax.set_title(f"Class-wise mean absolute SHAP values on {date}", fontsize=14, pad=12)
    ax.tick_params(labelsize=11)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    if x_max is None:
        ax.set_xlim(0, max(float(offsets.max()) * 1.06, 0.01))
    else:
        ax.set_xlim(0, x_max)
        ax.xaxis.set_major_locator(MultipleLocator(0.05))
    ax.legend(loc="lower right", fontsize=9)
    fig.subplots_adjust(left=0.26, right=0.96, bottom=0.12, top=0.90)

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"SHAP_class_stacked_{date}_{model_name}.png"
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    return path
