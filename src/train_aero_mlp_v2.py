from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from problem_v2_spec import AERO_INPUT_COLS, AERO_OUTPUT_COLS


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, default="data/prepared_aero_v2")
    p.add_argument("--outdir", type=str, default="models/aero_mlp_v2")
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cx-weight", type=float, default=3.0)
    p.add_argument("--k-weight", type=float, default=2.0)
    p.add_argument("--early-stop-patience", type=int, default=50)
    p.add_argument("--lr-patience", type=int, default=15)
    p.add_argument("--promote-variant", choices=["best_loss", "best_cx", "best_combo"], default="best_combo")
    return p.parse_args()


def load_split(data_dir: Path, split: str):
    x = np.load(data_dir / f"{split}_X.npy").astype(np.float32)
    y = np.load(data_dir / f"{split}_Y.npy").astype(np.float32)
    w = np.load(data_dir / f"{split}_sample_weight.npy").astype(np.float32)
    return x, y, w


def build_model(input_dim: int, output_dim: int, cx_weight: float, k_weight: float) -> keras.Model:
    x_in = keras.Input(shape=(input_dim,))
    x = x_in
    for i, width in enumerate([512, 256, 128, 64]):
        x = layers.Dense(width, activation="silu", name=f"dense_{i+1}")(x)
        x = layers.BatchNormalization(name=f"bn_{i+1}")(x)
        x = layers.Dropout(0.03, name=f"drop_{i+1}")(x)
    y_out = layers.Dense(output_dim, name="y")(x)
    model = keras.Model(x_in, y_out)
    weights = tf.constant([cx_weight, 1.0, 1.0, 1.0, 1.0, k_weight], dtype=tf.float32)

    def weighted_mse(y_true, y_pred):
        err = tf.square(y_true - y_pred) * weights
        return tf.reduce_mean(err, axis=-1)

    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=weighted_mse, metrics=[keras.metrics.MeanAbsoluteError()])
    return model


class ValCxCheckpoint(keras.callbacks.Callback):
    def __init__(self, x_val_s, y_val, sy: StandardScaler, outdir: Path):
        super().__init__()
        self.x_val_s = x_val_s
        self.y_val = y_val
        self.sy = sy
        self.outdir = outdir
        self.best_val_cx_mae = float("inf")
        self.best_val_combo = float("inf")
        self.best_epoch = 0
        self.best_combo_epoch = 0
        self.best_val_k_mae = float("inf")
        self.best_k_epoch = 0
        self.history: list[dict[str, float]] = []
        self.cx_scale = float(np.std(y_val[:, 0]) + 1e-12)
        self.cy_scale = float(np.std(y_val[:, 1]) + 1e-12)
        self.k_scale = float(np.std(y_val[:, 5]) + 1e-12)

    def on_epoch_end(self, epoch, logs=None):
        y_pred_s = self.model.predict(self.x_val_s, verbose=0)
        y_pred = self.sy.inverse_transform(y_pred_s)
        cx_mae = float(mean_absolute_error(self.y_val[:, 0], y_pred[:, 0]))
        cy_mae = float(mean_absolute_error(self.y_val[:, 1], y_pred[:, 1]))
        k_mae = float(mean_absolute_error(self.y_val[:, 5], y_pred[:, 5]))
        combo = 0.45 * (cx_mae / self.cx_scale) + 0.30 * (cy_mae / self.cy_scale) + 0.25 * (k_mae / self.k_scale)
        row = {
            "epoch": int(epoch + 1),
            "val_cx_mae": cx_mae,
            "val_cy_mae": cy_mae,
            "val_K_mae": k_mae,
            "val_combo_score": combo,
            "val_loss": float(logs.get("val_loss", float("nan"))) if logs else float("nan"),
            "val_mae": float(logs.get("val_mean_absolute_error", float("nan"))) if logs else float("nan"),
        }
        self.history.append(row)
        if cx_mae < self.best_val_cx_mae:
            self.best_val_cx_mae = cx_mae
            self.best_epoch = int(epoch + 1)
            self.model.save(self.outdir / "aero_mlp_v2_best_cx.keras")
        if combo < self.best_val_combo:
            self.best_val_combo = combo
            self.best_combo_epoch = int(epoch + 1)
            self.model.save(self.outdir / "aero_mlp_v2_best_combo.keras")
        if k_mae < self.best_val_k_mae:
            self.best_val_k_mae = k_mae
            self.best_k_epoch = int(epoch + 1)


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray):
    out = {}
    for i, name in enumerate(AERO_OUTPUT_COLS):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        out[name] = {
            "r2": float(r2_score(yt, yp)),
            "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
            "mae": float(mean_absolute_error(yt, yp)),
        }
    return out


def correlation_report(
    x_all: np.ndarray,
    y_all: np.ndarray,
    outdir: Path,
    top_n: int = 30,
) -> dict:
    all_cols = AERO_INPUT_COLS + AERO_OUTPUT_COLS
    df = pd.DataFrame(np.column_stack([x_all, y_all]), columns=all_cols)
    corr_all = df.corr(method="pearson")
    corr_outputs = corr_all.loc[AERO_OUTPUT_COLS, AERO_OUTPUT_COLS]
    corr_input_output = corr_all.loc[AERO_INPUT_COLS, AERO_OUTPUT_COLS]

    corr_all.to_csv(outdir / "data_correlation_all.csv")
    corr_outputs.to_csv(outdir / "data_correlation_outputs.csv")
    corr_input_output.to_csv(outdir / "data_correlation_input_output.csv")

    output_pairs = []
    for i, a in enumerate(AERO_OUTPUT_COLS):
        for b in AERO_OUTPUT_COLS[i + 1:]:
            val = corr_outputs.loc[a, b]
            if np.isfinite(val):
                output_pairs.append({"a": a, "b": b, "corr": float(val), "abs_corr": float(abs(val))})
    output_pairs.sort(key=lambda item: item["abs_corr"], reverse=True)

    input_output_pairs = []
    for inp in AERO_INPUT_COLS:
        for out in AERO_OUTPUT_COLS:
            val = corr_input_output.loc[inp, out]
            if np.isfinite(val):
                input_output_pairs.append({
                    "input": inp,
                    "output": out,
                    "corr": float(val),
                    "abs_corr": float(abs(val)),
                })
    input_output_pairs.sort(key=lambda item: item["abs_corr"], reverse=True)

    return {
        "files": {
            "all": "data_correlation_all.csv",
            "outputs": "data_correlation_outputs.csv",
            "input_output": "data_correlation_input_output.csv",
        },
        "top_output_output_abs_corr": output_pairs[:top_n],
        "top_input_output_abs_corr": input_output_pairs[:top_n],
    }


def main() -> None:
    args = parse_args()
    tf.keras.utils.set_random_seed(args.seed)
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir
    outdir = Path(args.outdir)
    if not outdir.is_absolute():
        outdir = PROJECT_ROOT / outdir
    outdir.mkdir(parents=True, exist_ok=True)

    x_train, y_train, w_train = load_split(data_dir, "train")
    x_val, y_val, w_val = load_split(data_dir, "val")
    x_test, y_test, _ = load_split(data_dir, "test")

    sx = StandardScaler()
    sy = StandardScaler()
    x_train_s = sx.fit_transform(x_train)
    x_val_s = sx.transform(x_val)
    x_test_s = sx.transform(x_test)
    y_train_s = sy.fit_transform(y_train)
    y_val_s = sy.transform(y_val)

    model = build_model(x_train.shape[1], y_train.shape[1], args.cx_weight, args.k_weight)
    cx_ckpt = ValCxCheckpoint(x_val_s, y_val, sy, outdir)
    cbs = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=args.early_stop_patience, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=args.lr_patience, min_lr=1e-6),
        keras.callbacks.ModelCheckpoint(outdir / "aero_mlp_v2_best_loss.keras", monitor="val_loss", save_best_only=True),
        cx_ckpt,
    ]
    hist = model.fit(
        x_train_s,
        y_train_s,
        validation_data=(x_val_s, y_val_s, w_val),
        sample_weight=w_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=2,
        callbacks=cbs,
    )
    if not (outdir / "aero_mlp_v2_best_cx.keras").exists():
        raise FileNotFoundError(f"Missing cx checkpoint in {outdir}")
    if not (outdir / "aero_mlp_v2_best_combo.keras").exists():
        raise FileNotFoundError(f"Missing combo checkpoint in {outdir}")
    promote_map = {
        "best_loss": "aero_mlp_v2_best_loss.keras",
        "best_cx": "aero_mlp_v2_best_cx.keras",
        "best_combo": "aero_mlp_v2_best_combo.keras",
    }
    promote_file = promote_map[args.promote_variant]
    keras.models.load_model(outdir / promote_file, compile=False).save(outdir / "aero_mlp_v2_best.keras")
    model = keras.models.load_model(outdir / promote_file, compile=False)

    def pred_back(x):
        return sy.inverse_transform(model.predict(x, verbose=0))

    pred_train = pred_back(x_train_s)
    pred_val = pred_back(x_val_s)
    pred_test = pred_back(x_test_s)

    joblib.dump(sx, outdir / "scaler_X.joblib")
    joblib.dump(sy, outdir / "scaler_Y.joblib")
    np.save(outdir / "pred_test.npy", pred_test)
    (outdir / "history.json").write_text(json.dumps(hist.history, indent=2), encoding="utf-8")
    (outdir / "val_cx_history.json").write_text(json.dumps(cx_ckpt.history, indent=2), encoding="utf-8")
    x_all = np.vstack([x_train, x_val, x_test])
    y_all = np.vstack([y_train, y_val, y_test])
    metrics = {
        "input_cols": AERO_INPUT_COLS,
        "output_cols": AERO_OUTPUT_COLS,
        "train_config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "cx_weight": args.cx_weight,
            "k_weight": args.k_weight,
            "early_stop_patience": args.early_stop_patience,
            "lr_patience": args.lr_patience,
            "promoted_variant": args.promote_variant,
            "best_val_cx_mae": cx_ckpt.best_val_cx_mae,
            "best_val_cx_epoch": cx_ckpt.best_epoch,
            "best_val_K_mae": cx_ckpt.best_val_k_mae,
            "best_val_K_epoch": cx_ckpt.best_k_epoch,
            "best_val_combo_score": cx_ckpt.best_val_combo,
            "best_val_combo_epoch": cx_ckpt.best_combo_epoch,
        },
        "data_correlation": correlation_report(x_all, y_all, outdir),
        "train_metrics": metric_dict(y_train, pred_train),
        "val_metrics": metric_dict(y_val, pred_val),
        "test_metrics": metric_dict(y_test, pred_test),
    }
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved aero model and metrics in {outdir}")


if __name__ == "__main__":
    main()
