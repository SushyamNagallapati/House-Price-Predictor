import os
import io
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# -----------------------------
# Config & utils
# -----------------------------
st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="wide")
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

FEATURE_ORDER = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms",
    "Population", "AveOccup", "Latitude", "Longitude"
]
TARGET_NAME = "MedHouseVal"

ARTI_DIR = "artifacts"
os.makedirs(ARTI_DIR, exist_ok=True)

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def metrics_df(y_true, y_pred, prefix=""):
    return pd.DataFrame({
        f"{prefix}RMSE": [rmse(y_true, y_pred)],
        f"{prefix}MAE": [mean_absolute_error(y_true, y_pred)],
        f"{prefix}R2": [r2_score(y_true, y_pred)]
    })

def load_default_data():
    data = fetch_california_housing(as_frame=True)
    df = data.frame.copy()
    X = df.drop(columns=[TARGET_NAME])
    y = df[TARGET_NAME]
    return X, y

def train_val_test_split(X, y, seed=RANDOM_SEED):
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=0.20, random_state=seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.25, random_state=seed
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

def build_pipeline(degree=2, alpha=1.0):
    # StandardScaler(with_mean=False) plays nice with sparse-ish PolyFeatures
    scaler = StandardScaler(with_mean=False) if degree > 1 else StandardScaler()
    return Pipeline([
        ("scaler", scaler),
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("ridge", Ridge(alpha=alpha, random_state=RANDOM_SEED))
    ])

def save_artifacts(pipeline, degree, alpha, cv_stats, test_stats, features):
    model_path = os.path.join(ARTI_DIR, f"house_price_ridge_deg{degree}_alpha{alpha}.joblib")
    joblib.dump(pipeline, model_path)

    summary = {
        "best_degree": int(degree),
        "best_alpha": float(alpha),
        "cv_rmse_mean": float(cv_stats["mean"]),
        "cv_rmse_std": float(cv_stats["std"]),
        "test_rmse": float(test_stats["RMSE"]),
        "test_mae": float(test_stats["MAE"]),
        "test_r2": float(test_stats["R2"]),
        "features": features
    }
    with open(os.path.join(ARTI_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return model_path

def validate_columns(df: pd.DataFrame):
    missing = [c for c in FEATURE_ORDER if c not in df.columns]
    extra = [c for c in df.columns if c not in FEATURE_ORDER]
    return missing, extra

def format_dollars(val_100k):
    # Target is in $100k; convert to dollars for display
    return f"${val_100k*100000:,.0f}"

# -----------------------------
# Sidebar: Mode
# -----------------------------
st.sidebar.header("⚙️ Settings")
mode = st.sidebar.radio(
    "Mode",
    ["Train model", "Load model & Predict"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.caption("Built with PolynomialFeatures and Ridge.")

# -----------------------------
# TRAIN MODE
# -----------------------------
if mode == "Train model":
    st.title("🏠 House Price Predictor")
    st.write("This trains a Ridge-regression pipeline with polynomial features on the California Housing dataset.")

    # Controls
    st.subheader("1) Hyperparameters")
    col1, col2 = st.columns(2)
    with col1:
        degree = st.slider("Polynomial degree", min_value=1, max_value=6, value=2, step=1)
    with col2:
        alphas_exp = st.slider("Alpha (λ) exponent range (10^x)", -4, 4, (-2, 2))
    alpha_grid = np.logspace(alphas_exp[0], alphas_exp[1], num=10)

    # Data
    st.subheader("2) Data")
    st.caption("Using scikit-learn California Housing dataset.")
    X, y = load_default_data()
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y)
    st.write(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # Sweep alphas
    st.subheader("3) Tune α by validation RMSE")
    rows = []
    for a in alpha_grid:
        pipe = build_pipeline(degree=degree, alpha=a)
        pipe.fit(X_train, y_train)
        y_tr = pipe.predict(X_train)
        y_va = pipe.predict(X_val)
        rows.append({
            "alpha": a,
            "train_rmse": rmse(y_train, y_tr),
            "val_rmse": rmse(y_val, y_va),
            "val_r2": r2_score(y_val, y_va)
        })
    sweep_df = pd.DataFrame(rows).sort_values("val_rmse")
    st.dataframe(sweep_df.style.format({"alpha": "{:.5g}", "train_rmse": "{:.3f}", "val_rmse": "{:.3f}", "val_r2": "{:.4f}"}), use_container_width=True)

    best_row = sweep_df.iloc[0]
    best_alpha = float(best_row["alpha"])
    st.success(f"Best α (by Validation RMSE): {best_alpha:g}  |  Val RMSE: {best_row['val_rmse']:.3f}")

    # CV (on train set only) for stability
    st.subheader("4) 5-fold CV on Train (sanity check)")
    cv_pipe = build_pipeline(degree=degree, alpha=best_alpha)
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    neg_mse_scores = cross_val_score(cv_pipe, X_train, y_train, scoring="neg_mean_squared_error", cv=kf)
    cv_rmse = np.sqrt(-neg_mse_scores)
    st.write(pd.DataFrame({"fold_rmse": cv_rmse}))
    cv_stats = {"mean": float(cv_rmse.mean()), "std": float(cv_rmse.std(ddof=1))}
    st.info(f"CV RMSE — mean: {cv_stats['mean']:.3f}, std: {cv_stats['std']:.3f}")

    # Final train on train+val, test once
    st.subheader("5) Final Train (Train+Val) and Test Evaluation")
    X_trval = pd.concat([X_train, X_val], axis=0)
    y_trval = pd.concat([y_train, y_val], axis=0)
    final_pipe = build_pipeline(degree=degree, alpha=best_alpha).fit(X_trval, y_trval)
    y_test_pred = final_pipe.predict(X_test)

    test_metrics = metrics_df(y_test, y_test_pred).iloc[0].to_dict()
    st.write(pd.DataFrame([test_metrics]).style.format({"RMSE": "{:.3f}", "MAE": "{:.3f}", "R2": "{:.4f}"}))

    # Save artifacts
    if st.button("💾 Save model & summary"):
        model_path = save_artifacts(
            final_pipe, degree, best_alpha,
            cv_stats=cv_stats,
            test_stats=test_metrics,
            features=FEATURE_ORDER
        )
        st.success(f"Saved model to: {model_path}")
        st.caption("Also saved: artifacts/summary.json")

    st.markdown("---")
    st.caption("Tip: Move to **Load model & Predict** to upload CSV and get predictions.")

# -----------------------------
# PREDICT MODE
# -----------------------------
else:
    st.title("🏠 House Price Predictor — Predict from CSV")
    st.write("Load a saved model and upload a CSV with the required columns to get house price predictions.")

    # Load model
    model_files = [f for f in os.listdir(ARTI_DIR) if f.endswith(".joblib")]
    if not model_files:
        st.warning("No saved models found in ./artifacts. Switch to **Train model** first, save a model, then come back.")
        st.stop()

    model_choice = st.selectbox("Select a saved model (.joblib):", sorted(model_files))
    model_path = os.path.join(ARTI_DIR, model_choice)
    pipe = joblib.load(model_path)
    st.success(f"Loaded model: {model_choice}")

    # Show expected columns
    with st.expander("Expected CSV format (columns & order)"):
        st.write("Columns must include (order will be aligned automatically):")
        st.code(",".join(FEATURE_ORDER), language="text")
        st.write("All values are numeric. Target is NOT required in upload.")
        st.caption("If your CSV has extra columns, we’ll ignore them. If required columns are missing, you’ll get a clear error.")

    # File uploader
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        try:
            df_raw = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            st.stop()

        st.subheader("1) Input preview")
        st.dataframe(df_raw.head())

        # Validate
        missing, extra = validate_columns(df_raw)
        if missing:
            st.error(f"Missing required columns: {missing}")
            st.stop()
        else:
            if extra:
                st.info(f"Ignoring extra columns: {extra}")
            # Align column order & subset
            X_infer = df_raw.reindex(columns=FEATURE_ORDER)

        # Predict
        if st.button("🔮 Predict"):
            preds_100k = pipe.predict(X_infer)
            out = df_raw.copy()
            out["Predicted_MedHouseVal_100k"] = preds_100k
            out["Predicted_MedHouseVal_$"] = [format_dollars(v) for v in preds_100k]

            st.subheader("2) Predictions")
            st.dataframe(out.head(50))

            # Download
            csv_buf = io.StringIO()
            out.to_csv(csv_buf, index=False)
            st.download_button(
                label="⬇️ Download predictions CSV",
                data=csv_buf.getvalue(),
                file_name="predictions.csv",
                mime="text/csv"
            )
