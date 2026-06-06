import gc
import time
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

try:
    import shap
except ImportError:
    shap = None

RANDOM_STATE = 42
BASE_DIR = Path(__file__).resolve().parent

sns.set_theme(style="whitegrid")

TRAINING_PROFILES = {
    "Quick demo": {
        "sample_size": 50_000,
        "cv_folds": 3,
        "tune_random_forest": False,
        "description": "Fastest option for classroom demos and UI checks.",
    },
    "Balanced demo": {
        "sample_size": 100_000,
        "cv_folds": 3,
        "tune_random_forest": False,
        "description": "Good balance between speed and reasonably stable metrics.",
    },
    "Proper training": {
        "sample_size": None,
        "cv_folds": 5,
        "tune_random_forest": True,
        "description": "Uses the full dataset, stronger cross-validation, and tuned Random Forest.",
    },
}

def find_dataset_path() -> Path | None:
    candidates = [
        BASE_DIR / "creditcard.csv",
        BASE_DIR / "creditcard.csv.zip",
        BASE_DIR / "creditcard.zip",
        BASE_DIR / "creditcard .csv",
        Path.cwd() / "creditcard.csv",
        Path.cwd() / "creditcard.csv.zip",
        Path.cwd() / "creditcard.zip",
        Path.cwd() / "creditcard .csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None

@st.cache_data(show_spinner=False)
def load_dataset(file_or_path) -> pd.DataFrame:
    # Handle UploadedFile and path objects that end in .zip
    name = getattr(file_or_path, "name", str(file_or_path))
    if name.endswith(".zip"):
        return pd.read_csv(file_or_path, compression="zip")
    return pd.read_csv(file_or_path)

def take_stratified_sample(
    df: pd.DataFrame, sample_size: int | None, random_state: int = RANDOM_STATE
) -> pd.DataFrame:
    if sample_size is None or sample_size >= len(df):
        return df.copy()

    sampled_index, _ = train_test_split(
        df.index,
        train_size=sample_size,
        stratify=df["Class"],
        random_state=random_state,
    )
    return df.loc[sampled_index].copy().reset_index(drop=True)

def build_pipelines(smote_strategy: float) -> dict[str, ImbPipeline]:
    return {
        "Logistic Regression": ImbPipeline(
            [
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=0.95, random_state=RANDOM_STATE)),
                ("smote", SMOTE(random_state=RANDOM_STATE, sampling_strategy=smote_strategy)),
                ("model", LogisticRegression(max_iter=3000, random_state=RANDOM_STATE)),
            ]
        ),
        "Random Forest": ImbPipeline(
            [
                ("smote", SMOTE(random_state=RANDOM_STATE, sampling_strategy=smote_strategy)),
                ("model", RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=1, class_weight="balanced_subsample")),
            ]
        ),
        "Gradient Boosting": ImbPipeline(
            [
                ("smote", SMOTE(random_state=RANDOM_STATE, sampling_strategy=smote_strategy)),
                ("model", GradientBoostingClassifier(n_estimators=50, random_state=RANDOM_STATE)),
            ]
        ),
        "Support Vector Machine": ImbPipeline(
            [
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=0.95, random_state=RANDOM_STATE)),
                ("smote", SMOTE(random_state=RANDOM_STATE, sampling_strategy=smote_strategy)),
                ("model", LinearSVC(random_state=RANDOM_STATE, max_iter=5000)),
            ]
        ),
    }

def get_model_scores(pipeline: ImbPipeline, features: pd.DataFrame) -> np.ndarray:
    if hasattr(pipeline, "predict_proba"):
        return pipeline.predict_proba(features)[:, 1]
    if hasattr(pipeline, "decision_function"):
        return pipeline.decision_function(features)
    raise AttributeError("The selected model does not expose scores for ROC-AUC.")


def get_risk_score_details(pipeline: ImbPipeline, features: pd.DataFrame) -> tuple[float, str]:
    if hasattr(pipeline, "predict_proba"):
        probability = float(pipeline.predict_proba(features)[0, 1])
        return probability, "Fraud probability"
    if hasattr(pipeline, "decision_function"):
        decision_score = float(np.ravel(pipeline.decision_function(features))[0])
        normalized_score = 1.0 / (1.0 + np.exp(-np.clip(decision_score, -500, 500)))
        return normalized_score, "Normalized decision score (not a calibrated probability)"
    raise AttributeError("The selected model does not expose a usable prediction score.")


def get_feature_importance_df(
    pipeline: ImbPipeline, feature_names: pd.Index
) -> pd.DataFrame | None:
    model = pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return None

    importances = np.asarray(model.feature_importances_)
    if len(importances) != len(feature_names):
        return None

    return pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": importances,
        }
    ).sort_values(by="Importance", ascending=False)

def is_pca_pipeline(pipeline: ImbPipeline) -> bool:
    return "pca" in pipeline.named_steps

def supports_tree_importance(pipeline: ImbPipeline) -> bool:
    model = pipeline.named_steps["model"]
    return hasattr(model, "feature_importances_") and not is_pca_pipeline(pipeline)

def get_explainability_feature_frame(results: dict, dataset: pd.DataFrame) -> pd.DataFrame:
    feature_columns = results["test_samples"].drop(columns=["Actual Class"]).columns
    if set(feature_columns).issubset(dataset.columns):
        return dataset.loc[:, feature_columns].copy()
    return results["test_samples"].drop(columns=["Actual Class"]).copy()

def build_feature_importance_chart(
    feature_importance_df: pd.DataFrame,
    model_name: str,
    top_n: int = 10,
):
    top_features = feature_importance_df.head(top_n).sort_values("Importance")
    fig = px.bar(
        top_features,
        x="Importance",
        y="Feature",
        orientation="h",
        title=f"{model_name}: Top {top_n} Feature Importance",
        hover_data={
            "Feature": True,
            "Importance": ":.6f",
        },
        labels={
            "Importance": "Model-specific importance score",
            "Feature": "Feature",
        },
        color="Importance",
        color_continuous_scale="Viridis",
    )
    fig.update_layout(
        height=460,
        margin=dict(l=10, r=10, t=55, b=20),
        coloraxis_showscale=False,
    )
    return fig

def compute_shap_values(pipeline: ImbPipeline, features: pd.DataFrame):
    if shap is None:
        return None, "SHAP is not installed in this environment."
    if not supports_tree_importance(pipeline):
        return None, "SHAP explanations are available here only for non-PCA tree-based models."

    model = pipeline.named_steps["model"]
    sample_size = min(len(features), 1000)
    background = features.sample(sample_size, random_state=RANDOM_STATE)
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(background)
    except Exception as exc:
        return None, f"SHAP could not explain this model: {exc}"

    if isinstance(shap_values, list):
        shap_values = shap_values[-1]
    if getattr(shap_values, "ndim", 2) == 3:
        shap_values = shap_values[:, :, -1]

    return {"values": np.asarray(shap_values), "features": background}, None

def build_shap_global_chart(shap_result: dict, model_name: str, top_n: int = 10):
    shap_values = shap_result["values"]
    features = shap_result["features"]
    mean_abs = np.abs(shap_values).mean(axis=0)
    shap_df = pd.DataFrame(
        {"Feature": features.columns, "Mean |SHAP value|": mean_abs}
    ).sort_values("Mean |SHAP value|", ascending=False)
    top_features = shap_df.head(top_n).sort_values("Mean |SHAP value|")

    fig = px.bar(
        top_features,
        x="Mean |SHAP value|",
        y="Feature",
        orientation="h",
        title=f"{model_name}: Top {top_n} SHAP Drivers",
        hover_data={"Feature": True, "Mean |SHAP value|": ":.6f"},
        labels={"Mean |SHAP value|": "Mean absolute SHAP value"},
        color="Mean |SHAP value|",
        color_continuous_scale="Teal",
    )
    fig.update_layout(
        height=460,
        margin=dict(l=10, r=10, t=55, b=20),
        coloraxis_showscale=False,
    )
    return fig, shap_df

def build_local_shap_chart(
    pipeline: ImbPipeline,
    feature_row: pd.DataFrame,
    model_name: str,
    top_n: int = 10,
):
    if shap is None:
        return None, "SHAP is not installed in this environment."
    if not supports_tree_importance(pipeline):
        return None, "Local SHAP explanations are available here only for non-PCA tree-based models."

    model = pipeline.named_steps["model"]
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(feature_row)
    except Exception as exc:
        return None, f"SHAP could not explain this transaction: {exc}"

    if isinstance(shap_values, list):
        shap_values = shap_values[-1]
    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, -1]
    row_values = shap_values[0]

    local_df = pd.DataFrame(
        {
            "Feature": feature_row.columns,
            "SHAP value": row_values,
            "Feature value": feature_row.iloc[0].values,
            "Impact direction": np.where(row_values >= 0, "Increases fraud score", "Decreases fraud score"),
        }
    )
    local_df["Absolute impact"] = local_df["SHAP value"].abs()
    local_df = local_df.sort_values("Absolute impact", ascending=False)
    chart_df = local_df.head(top_n).sort_values("Absolute impact")

    fig = px.bar(
        chart_df,
        x="SHAP value",
        y="Feature",
        orientation="h",
        title=f"{model_name}: Local SHAP Explanation",
        hover_data={
            "Feature": True,
            "Feature value": ":.6f",
            "SHAP value": ":.6f",
            "Impact direction": True,
            "Absolute impact": False,
        },
        color="Impact direction",
        color_discrete_map={
            "Increases fraud score": "#DD8452",
            "Decreases fraud score": "#4C72B0",
        },
    )
    fig.update_layout(height=460, margin=dict(l=10, r=10, t=55, b=20))
    return fig, local_df

def evaluate_model(
    model_name: str,
    pipeline: ImbPipeline,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[dict[str, float | str], np.ndarray, np.ndarray]:
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)
    scores = get_model_scores(pipeline, x_test)

    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1-score": f1_score(y_test, predictions, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, scores),
    }
    return metrics, predictions, scores

@st.cache_resource(show_spinner=False)
def run_experiment(
    _df: pd.DataFrame,
    sample_size: int | None,
    smote_strategy: float,
    cv_folds: int,
    tune_random_forest: bool,
) -> dict:
    working_df = take_stratified_sample(_df, sample_size)

    x = working_df.drop(columns="Class")
    y = working_df["Class"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    pipelines = build_pipelines(smote_strategy)

    cv_rows = []
    for model_name, pipeline in pipelines.items():
        scores = cross_validate(pipeline, x_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
        cv_rows.append(
            {
                "Model": model_name,
                "CV Accuracy": scores["test_accuracy"].mean(),
                "CV Precision": scores["test_precision"].mean(),
                "CV Recall": scores["test_recall"].mean(),
                "CV F1": scores["test_f1"].mean(),
                "CV ROC-AUC": scores["test_roc_auc"].mean(),
            }
        )

    tuning_summary = None
    if tune_random_forest:
        param_grid = {
            "model__n_estimators": [100, 200],
            "model__max_depth": [None, 16],
            "model__min_samples_split": [2, 5],
        }
        search = GridSearchCV(
            estimator=pipelines["Random Forest"],
            param_grid=param_grid,
            scoring="recall",
            cv=cv,
            n_jobs=1,
        )
        search.fit(x_train, y_train)
        pipelines["Random Forest"] = search.best_estimator_
        tuning_summary = {
            "best_params": search.best_params_,
            "best_cv_recall": search.best_score_,
        }

    comparison_rows = []
    model_outputs = {}
    for model_name, pipeline in pipelines.items():
        metrics, predictions, scores = evaluate_model(
            model_name, pipeline, x_train, x_test, y_train, y_test
        )
        feature_importance_df = get_feature_importance_df(pipeline, x_train.columns)
        comparison_rows.append(metrics)
        model_outputs[model_name] = {
            "pipeline": pipeline,
            "predictions": predictions,
            "scores": scores,
            "confusion_matrix": confusion_matrix(y_test, predictions),
            "feature_importance_df": feature_importance_df.reset_index(drop=True)
            if feature_importance_df is not None
            else None,
        }

    comparison_df = pd.DataFrame(comparison_rows).sort_values(by=["Recall", "ROC-AUC"], ascending=False)
    cv_df = pd.DataFrame(cv_rows).sort_values(by="CV Recall", ascending=False)

    feature_importance_df = model_outputs["Random Forest"]["feature_importance_df"]

    test_samples = x_test.copy()
    test_samples["Actual Class"] = y_test.values
    test_samples = test_samples.reset_index(drop=True)

    result = {
        "comparison_df": comparison_df.reset_index(drop=True),
        "cv_df": cv_df.reset_index(drop=True),
        "model_outputs": model_outputs,
        "feature_importance_df": feature_importance_df.reset_index(drop=True),
        "train_samples": x_train.reset_index(drop=True),
        "test_samples": test_samples,
        "y_test": y_test.reset_index(drop=True),
        "tuning_summary": tuning_summary,
        "dataset_summary": {
            "rows_used": len(working_df),
            "train_rows": len(x_train),
            "test_rows": len(x_test),
            "fraud_cases_used": int(working_df["Class"].sum()),
            "cv_folds": cv_folds,
        },
    }
    # Free the large intermediate DataFrame to reduce memory pressure
    del working_df, x_train, x_test, y_train
    gc.collect()
    return result

def plot_class_distribution(df: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    class_percent = df["Class"].value_counts(normalize=True).sort_index() * 100
    sns.countplot(x="Class", data=df, ax=axes[0], palette="Set2")
    axes[0].set_title("Class Distribution")
    class_percent.plot(kind="bar", ax=axes[1], color=["#4C72B0", "#DD8452"])
    axes[1].set_title("Class Percentage")
    axes[1].tick_params(axis="x", rotation=0)
    fig.tight_layout()
    return fig

def plot_confusion_matrices(results: dict) -> plt.Figure:
    num_models = len(results["model_outputs"])
    fig, axes = plt.subplots(1, num_models, figsize=(5 * num_models, 4))
    for axis, (model_name, output) in zip(axes, results["model_outputs"].items()):
        sns.heatmap(output["confusion_matrix"], annot=True, fmt="d", cmap="Blues", cbar=False, ax=axis)
        axis.set_title(f"{model_name}")
        axis.set_xlabel("Predicted")
        axis.set_ylabel("Actual")
    fig.tight_layout()
    return fig

def plot_roc_curves(results: dict) -> plt.Figure:
    fig, axis = plt.subplots(figsize=(8, 6))
    y_test = results["y_test"]
    for model_name, output in results["model_outputs"].items():
        fpr, tpr, _ = roc_curve(y_test, output["scores"])
        auc_score = roc_auc_score(y_test, output["scores"])
        axis.plot(fpr, tpr, linewidth=2, label=f"{model_name} (AUC = {auc_score:.4f})")
    axis.plot([0, 1], [0, 1], linestyle="--", color="black", label="Random Guess")
    axis.set_title("ROC Curve Comparison")
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.legend()
    fig.tight_layout()
    return fig

def plot_local_explanation(
    feature_row: pd.DataFrame,
    test_samples: pd.DataFrame,
    feature_importances: pd.DataFrame,
    model_name: str,
) -> plt.Figure:
    top_features = feature_importances.head(5)["Feature"].tolist()
    
    # Calculate means for fraud and non-fraud
    legit_means = test_samples[test_samples["Actual Class"] == 0][top_features].mean()
    fraud_means = test_samples[test_samples["Actual Class"] == 1][top_features].mean()
    transaction_vals = feature_row[top_features].iloc[0]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(top_features))
    width = 0.25
    
    ax.bar(x - width, legit_means, width, label='Avg Legitimate', color='#4C72B0', alpha=0.8)
    ax.bar(x, fraud_means, width, label='Avg Fraud', color='#DD8452', alpha=0.8)
    ax.bar(x + width, transaction_vals, width, label='This Transaction', color='#55A868', edgecolor='black', linewidth=1.5)
    
    ax.set_title(f"{model_name}: Top 5 Influential Features")
    ax.set_xticks(x)
    ax.set_xticklabels(top_features)
    ax.legend()
    fig.tight_layout()
    return fig


# Streamlit Pages
def page_model_training():
    st.header("⚙️ Model Training & Global Performance")
    results = st.session_state.get("results")
    if not results:
        st.warning("Please run the training pipeline from the sidebar first.")
        return

    st.subheader("Model Comparison Table")
    st.dataframe(
        results["comparison_df"].style.format(
            {"Accuracy": "{:.4f}", "Precision": "{:.4f}", "Recall": "{:.4f}", "F1-score": "{:.4f}", "ROC-AUC": "{:.4f}"}
        ),
        use_container_width=True,
    )

    st.subheader("Confusion Matrices")
    st.pyplot(plot_confusion_matrices(results))

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("ROC Curves")
        st.pyplot(plot_roc_curves(results))
    with col2:
        st.subheader("Global Feature Importance (Random Forest)")
        top_features = results["feature_importance_df"].head(10)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(data=top_features, x="Importance", y="Feature", palette="viridis", ax=ax)
        st.pyplot(fig)


def page_manual_testing():
    st.header("🧪 Manual Testing Lab & Explainability")
    results = st.session_state.get("results")
    if not results:
        st.warning("Please run the training pipeline first.")
        return

    st.write("Test individual transactions and compare how different models evaluate them.")

    test_samples = results["test_samples"]
    
    col_input1, col_input2 = st.columns([1, 2])
    with col_input1:
        st.subheader("Input Data")
        input_method = st.radio("Choose Input Method", ["Select from Test Set", "Synthetic Transaction"])
        
        if input_method == "Select from Test Set":
            sample_index = st.slider("Select a test transaction ID", 0, len(test_samples) - 1, 0)
            selected_row = test_samples.iloc[sample_index].copy()
            actual_class = int(selected_row["Actual Class"])
            feature_row = selected_row.drop(labels=["Actual Class"]).to_frame().T
            st.info(f"Actual Class: **{'Fraud' if actual_class == 1 else 'Legitimate'}**")
        else:
            profile_type = st.selectbox(
                "Start synthetic data from",
                ["Neutral / Zero Features", "Legitimate-like Template", "Fraud-like Template"],
            )

            if "synthetic_v_features" not in st.session_state:
                st.session_state["synthetic_v_features"] = {f"V{i}": 0.0 for i in range(1, 29)}

            if st.button("Generate Synthetic Template"):
                if profile_type == "Neutral / Zero Features":
                    template_values = {f"V{i}": 0.0 for i in range(1, 29)}
                    st.session_state["synthetic_amount"] = 150.0
                    st.session_state["synthetic_time"] = 10000.0
                    st.session_state["synthetic_amount_input"] = 150.0
                    st.session_state["synthetic_time_input"] = 10000.0
                else:
                    target_cls = 0 if profile_type == "Legitimate-like Template" else 1
                    template_row = test_samples[test_samples["Actual Class"] == target_cls].sample(1).iloc[0]
                    template_values = template_row.drop(["Time", "Amount", "Actual Class"]).to_dict()
                    st.session_state["synthetic_amount"] = float(template_row["Amount"])
                    st.session_state["synthetic_time"] = float(template_row["Time"])
                    st.session_state["synthetic_amount_input"] = float(template_row["Amount"])
                    st.session_state["synthetic_time_input"] = float(template_row["Time"])

                st.session_state["synthetic_v_features"] = template_values
                for feature, value in template_values.items():
                    st.session_state[f"synthetic_{feature}"] = float(value)

            amt = st.number_input(
                "Transaction Amount ($)",
                min_value=0.0,
                value=float(st.session_state.get("synthetic_amount", 150.0)),
                key="synthetic_amount_input",
            )
            time_val = st.number_input(
                "Time (seconds since start)",
                min_value=0.0,
                value=float(st.session_state.get("synthetic_time", 10000.0)),
                key="synthetic_time_input",
            )

            v_features = st.session_state["synthetic_v_features"].copy()
            with st.expander("Advanced: edit V1-V28 synthetic features"):
                feature_columns = st.columns(4)
                for index in range(1, 29):
                    feature = f"V{index}"
                    with feature_columns[(index - 1) % 4]:
                        v_features[feature] = st.number_input(
                            feature,
                            value=float(v_features.get(feature, 0.0)),
                            format="%.6f",
                            key=f"synthetic_{feature}",
                        )

            feature_dict = {"Time": time_val}
            feature_dict.update(v_features)
            feature_dict["Amount"] = amt
            
            feature_row = pd.DataFrame([feature_dict])
            # Reorder columns to match training
            feature_row = feature_row[test_samples.drop(columns=["Actual Class"]).columns]
            st.info("Synthetic user-defined transaction ready for prediction.")

    with col_input2:
        st.subheader("Model Prediction")
        selected_model = st.selectbox("Compare with Model", list(results["model_outputs"].keys()))
        selected_output = results["model_outputs"][selected_model]
        pipeline = selected_output["pipeline"]
        
        # Make prediction
        prediction = int(pipeline.predict(feature_row)[0])
        risk_score_value, risk_score_label = get_risk_score_details(pipeline, feature_row)

        st.metric("Model Verdict", "🚨 FRAUD DETECTED" if prediction == 1 else "✅ LEGITIMATE")
        st.metric("Suspicion / Risk Score", f"{risk_score_value:.2%}")
        st.progress(risk_score_value)
        st.caption(risk_score_label)

        feature_importance_df = selected_output["feature_importance_df"]
        if feature_importance_df is None:
            st.subheader("Explanation Availability")
            st.info(
                "This chart is available only for tree-based models in the current app. "
                "Logistic Regression and SVM use PCA-transformed features, so their scores "
                "cannot be mapped back to the original columns with this simple explanation view."
            )
        else:
            st.subheader("Feature Importance Explanation")
            st.write(
                "Visual breakdown of how this model's top 5 influential features compare "
                "against typical legitimate and fraud transactions."
            )
            fig_explain = plot_local_explanation(
                feature_row, test_samples, feature_importance_df, selected_model
            )
            st.pyplot(fig_explain)


def page_model_explainability(dataset: pd.DataFrame):
    st.header("Model Explainability")
    results = st.session_state.get("results")
    if not results:
        st.warning("Please run the training pipeline first.")
        return

    st.write("Review global model drivers and transaction-level explanations for supported models.")

    model_names = list(results["model_outputs"].keys())
    selected_model = st.selectbox("Model", model_names, key="explainability_model")
    selected_output = results["model_outputs"][selected_model]
    pipeline = selected_output["pipeline"]
    feature_importance_df = selected_output["feature_importance_df"]

    if is_pca_pipeline(pipeline) or feature_importance_df is None:
        st.subheader("Explanation Availability")
        st.info(
            "This chart is available only for tree-based models in the current app. "
            "Logistic Regression and SVM use PCA-transformed features, so their scores "
            "cannot be mapped back to the original columns with this simple explanation view."
        )
        return

    with st.spinner("Preparing model explainability charts..."):
        feature_frame = get_explainability_feature_frame(results, dataset)

        st.subheader("Global Feature Importance")
        st.caption(
            "Scores are read from the fitted model's native feature importance values. "
            "Higher scores indicate features used more heavily by this model."
        )
        st.plotly_chart(
            build_feature_importance_chart(feature_importance_df, selected_model),
            use_container_width=True,
        )

        top_10 = feature_importance_df.head(10).copy()
        top_10.insert(0, "Rank", np.arange(1, len(top_10) + 1))
        st.subheader("Top 10 Most Important Features")
        st.dataframe(
            top_10.style.format({"Importance": "{:.6f}"}),
            use_container_width=True,
            hide_index=True,
        )

        with st.expander("Complete Ranked Feature List"):
            ranked_features = feature_importance_df.copy()
            ranked_features.insert(0, "Rank", np.arange(1, len(ranked_features) + 1))
            st.dataframe(
                ranked_features.style.format({"Importance": "{:.6f}"}),
                use_container_width=True,
                hide_index=True,
            )

    st.divider()
    st.subheader("SHAP Explainability")
    st.caption(
        "SHAP estimates how features contribute to model output. It is shown only for supported "
        "tree-based models and never used to change predictions."
    )

    shap_scope = st.radio(
        "SHAP data source",
        ["Uploaded/current dataset sample", "Training data", "Held-out test data"],
        horizontal=True,
    )
    if shap_scope == "Uploaded/current dataset sample":
        shap_features = feature_frame
    elif shap_scope == "Training data":
        shap_features = results.get(
            "train_samples",
            results["test_samples"].drop(columns=["Actual Class"]),
        )
    else:
        shap_features = results["test_samples"].drop(columns=["Actual Class"])

    with st.spinner("Computing global SHAP values..."):
        shap_result, shap_error = compute_shap_values(pipeline, shap_features)

    if shap_error:
        st.info(shap_error)
    else:
        shap_fig, shap_df = build_shap_global_chart(shap_result, selected_model)
        st.plotly_chart(shap_fig, use_container_width=True)
        with st.expander("Complete SHAP Ranked Feature List"):
            shap_ranked = shap_df.copy()
            shap_ranked.insert(0, "Rank", np.arange(1, len(shap_ranked) + 1))
            st.dataframe(
                shap_ranked.style.format({"Mean |SHAP value|": "{:.6f}"}),
                use_container_width=True,
                hide_index=True,
            )

    st.divider()
    st.subheader("Local Transaction Explanation")
    test_samples = results["test_samples"]
    sample_index = st.slider(
        "Select a held-out test transaction",
        0,
        len(test_samples) - 1,
        0,
        key="explainability_sample_index",
    )
    selected_row = test_samples.iloc[sample_index].copy()
    actual_class = int(selected_row["Actual Class"])
    feature_row = selected_row.drop(labels=["Actual Class"]).to_frame().T
    prediction = int(pipeline.predict(feature_row)[0])
    risk_score_value, risk_score_label = get_risk_score_details(pipeline, feature_row)

    metric_cols = st.columns(3)
    metric_cols[0].metric("Actual Class", "Fraud" if actual_class == 1 else "Legitimate")
    metric_cols[1].metric("Model Verdict", "Fraud" if prediction == 1 else "Legitimate")
    metric_cols[2].metric(risk_score_label, f"{risk_score_value:.2%}")
    



def page_admin_dashboard():
    st.header("🔄 Live Admin Dashboard & Simulation")
    results = st.session_state.get("results")
    if not results:
        st.warning("Please run the training pipeline first.")
        return

    st.write("Simulate real-time incoming transactions and monitor system metrics.")
    
    # Initialize simulation state
    if "sim_log" not in st.session_state:
        st.session_state["sim_log"] = []
    if "sim_log_model" not in st.session_state:
        st.session_state["sim_log_model"] = None
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        selected_model = st.selectbox("Active Monitoring Model", list(results["model_outputs"].keys()))
        if st.session_state["sim_log_model"] not in (None, selected_model):
            st.session_state["sim_log"] = []
            st.session_state["sim_log_model"] = selected_model
            st.info("Simulation history was cleared because the active model changed.")
        sim_speed = st.slider("Simulation Speed (sec/tx)", 0.1, 2.0, 0.5)
        num_sim = st.number_input("Transactions to simulate", 10, 500, 50)
        
        if st.button("🚀 Start Live Simulation", use_container_width=True):
            test_samples = results["test_samples"]
            pipeline = results["model_outputs"][selected_model]["pipeline"]
            st.session_state["sim_log_model"] = selected_model
            score_metric_label = (
                "Avg Fraud Probability"
                if hasattr(pipeline, "predict_proba")
                else "Avg Normalized Decision Score"
            )
            threshold_label = (
                "Fraud Threshold"
                if hasattr(pipeline, "predict_proba")
                else "Decision Threshold"
            )
            chart_ylabel = (
                "Fraud Probability"
                if hasattr(pipeline, "predict_proba")
                else "Normalized Decision Score"
            )
            
            metric_placeholder = st.empty()
            chart_placeholder = st.empty()
            
            # Simulate
            for i in range(num_sim):
                # Pick a random sample
                row = test_samples.sample(1).iloc[0]
                feature_row = row.drop("Actual Class").to_frame().T
                
                prediction = int(pipeline.predict(feature_row)[0])
                risk_score_value, risk_score_label = get_risk_score_details(pipeline, feature_row)
                
                log_entry = {
                    "Tx_ID": len(st.session_state["sim_log"]) + 1,
                    "Amount": row["Amount"],
                    "Risk_Score": risk_score_value,
                    "Risk_Score_Label": risk_score_label,
                    "Flagged": prediction == 1,
                    "Actual_Fraud": row["Actual Class"] == 1,
                }
                st.session_state["sim_log"].append(log_entry)
                
                # Update UI inside the loop
                log_df = pd.DataFrame(st.session_state["sim_log"])
                total_tx = len(log_df)
                total_flagged = log_df["Flagged"].sum()
                fraud_percent = (total_flagged / total_tx) * 100
                
                with metric_placeholder.container():
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Total Checked", total_tx)
                    m2.metric("Flagged Fraud", total_flagged)
                    m3.metric("Flag Rate", f"{fraud_percent:.1f}%")
                    m4.metric(score_metric_label, f"{log_df['Risk_Score'].mean():.2f}")
                
                with chart_placeholder.container():
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(log_df["Tx_ID"].tail(50), log_df["Risk_Score"].tail(50), color='orange', marker='o', markersize=4)
                    ax.axhline(y=0.5, color='r', linestyle='--', label=threshold_label)
                    ax.set_ylim(0, 1)
                    ax.set_title("Live Risk Score Trend (Last 50 Transactions)")
                    ax.set_ylabel(chart_ylabel)
                    ax.set_xlabel("Transaction ID")
                    ax.legend()
                    st.pyplot(fig)
                
                time.sleep(sim_speed)
                
        if st.button("Clear History", use_container_width=True):
            st.session_state["sim_log"] = []
            st.session_state["sim_log_model"] = selected_model
            st.rerun()

    with col2:
        st.subheader("System Metrics")
        if len(st.session_state["sim_log"]) > 0:
            log_df = pd.DataFrame(st.session_state["sim_log"])
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Checked", len(log_df))
            m2.metric("Flagged Fraud", log_df["Flagged"].sum())
            m3.metric("Flag Rate", f"{(log_df['Flagged'].sum() / len(log_df)) * 100:.1f}%")
            latest_score_label = log_df["Risk_Score_Label"].iloc[-1]
            avg_score_label = (
                "Avg Fraud Probability"
                if latest_score_label == "Fraud probability"
                else "Avg Normalized Decision Score"
            )
            threshold_label = (
                "Fraud Threshold"
                if latest_score_label == "Fraud probability"
                else "Decision Threshold"
            )
            chart_ylabel = (
                "Fraud Probability"
                if latest_score_label == "Fraud probability"
                else "Normalized Decision Score"
            )
            m4.metric(avg_score_label, f"{log_df['Risk_Score'].mean():.2f}")
            
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(log_df["Tx_ID"], log_df["Risk_Score"], color='orange', alpha=0.7)
            ax.axhline(y=0.5, color='r', linestyle='--', label=threshold_label)
            ax.set_ylim(0, 1)
            ax.set_title("Overall Risk Score Trend")
            ax.set_ylabel(chart_ylabel)
            ax.legend()
            st.pyplot(fig)
            
            st.dataframe(log_df.tail(10).iloc[::-1], use_container_width=True)
        else:
            st.info("No data yet. Click 'Start Live Simulation' to begin.")


def main() -> None:
    st.set_page_config(page_title="Credit Card Intelligence", page_icon="💳", layout="wide")
    
    st.sidebar.title("💳 Navigation")
    page = st.sidebar.radio("Go to", ["Model Training", "Manual Testing Lab", "Model Explainability", "Admin Dashboard"])

    dataset_path = find_dataset_path()

    st.sidebar.divider()
    st.sidebar.header("Data & Training Setup")
    
    if dataset_path is None:
        st.sidebar.warning(
            "Dataset not found in the app folder. Upload `creditcard.csv` or `creditcard.csv.zip` below."
        )
        uploaded_file = st.sidebar.file_uploader(
            "Upload creditcard.csv or creditcard.csv.zip", type=["csv", "zip"]
        )
        if uploaded_file is None:
            st.info(
                "Please upload `creditcard.csv` or `creditcard.csv.zip` in the sidebar to continue."
            )
            st.stop()
        dataset = load_dataset(uploaded_file)
    else:
        dataset = load_dataset(str(dataset_path))
    
    training_profile = st.sidebar.selectbox("Training profile", list(TRAINING_PROFILES.keys()), index=0)
    profile_config = TRAINING_PROFILES[training_profile]
    smote_strategy = st.sidebar.slider("SMOTE sampling strategy", 0.10, 1.00, 0.25, 0.05)
    
    if st.sidebar.button("Train Models", type="primary", use_container_width=True):
        try:
            with st.spinner("Training models, applying SMOTE, and computing metrics..."):
                results = run_experiment(
                    dataset,
                    profile_config["sample_size"],
                    smote_strategy,
                    profile_config["cv_folds"],
                    profile_config["tune_random_forest"],
                )
                st.session_state["results"] = results
                gc.collect()
                st.success("Training Complete!")
        except Exception:
            st.error(
                "Training failed (the server may have run out of memory). "
                "Try the **Quick demo** profile or a smaller SMOTE strategy."
            )
            st.code(traceback.format_exc())

    if page == "Model Training":
        st.title("Credit Card Fraud Models")
        st.write("Analyze dataset imbalance and train multiple machine learning models.")
        st.pyplot(plot_class_distribution(dataset))
        page_model_training()
    elif page == "Manual Testing Lab":
        page_manual_testing()
    elif page == "Model Explainability":
        page_model_explainability(dataset)
    elif page == "Admin Dashboard":
        page_admin_dashboard()

if __name__ == "__main__":
    main()
