from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from problem_v2_spec import INPUT_COLS_V2, OUTPUT_BALANCE_COLS


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, default="data/prepared_balance_v2")
    p.add_argument("--outdir", type=str, default="models/balance_mlp_v2")
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_split(data_dir: Path, split: str):
    x = np.load(data_dir / f"{split}_X.npy").astype(np.float32)
    y = np.load(data_dir / f"{split}_Y.npy").astype(np.float32)
    w = np.load(data_dir / f"{split}_sample_weight.npy").astype(np.float32)
    return x, y, w


def build_model(input_dim: int, output_dim: int) -> keras.Model:
    x_in = keras.Input(shape=(input_dim,))
    x = x_in
    for i, width in enumerate([256, 256, 128, 64]):
        x = layers.Dense(width, activation="silu", name=f"dense_{i+1}")(x)
        x = layers.BatchNormalization(name=f"bn_{i+1}")(x)
        x = layers.Dropout(0.03, name=f"drop_{i+1}")(x)
    y_out = layers.Dense(output_dim, name="y")(x)
    model = keras.Model(x_in, y_out)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse", metrics=[keras.metrics.MeanAbsoluteError()])
    return model


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray):
    out = {}
    for i, name in enumerate(OUTPUT_BALANCE_COLS):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        out[name] = {
            "r2": float(r2_score(yt, yp)),
            "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
            "mae": float(mean_absolute_error(yt, yp)),
        }
    return out


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

    model = build_model(x_train.shape[1], y_train.shape[1])
    cbs = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=40, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=15, min_lr=1e-6),
        keras.callbacks.ModelCheckpoint(outdir / "balance_mlp_v2_best.keras", monitor="val_loss", save_best_only=True),
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
    model = keras.models.load_model(outdir / "balance_mlp_v2_best.keras", compile=False)

    def pred_back(x):
        return sy.inverse_transform(model.predict(x, verbose=0))

    pred_train = pred_back(x_train_s)
    pred_val = pred_back(x_val_s)
    pred_test = pred_back(x_test_s)

    joblib.dump(sx, outdir / "scaler_X.joblib")
    joblib.dump(sy, outdir / "scaler_Y.joblib")
    np.save(outdir / "pred_test.npy", pred_test)
    (outdir / "history.json").write_text(json.dumps(hist.history, indent=2), encoding="utf-8")
    metrics = {
        "input_cols": INPUT_COLS_V2,
        "output_cols": OUTPUT_BALANCE_COLS,
        "train_metrics": metric_dict(y_train, pred_train),
        "val_metrics": metric_dict(y_val, pred_val),
        "test_metrics": metric_dict(y_test, pred_test),
    }
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved model and metrics in {outdir}")


if __name__ == "__main__":
    main()
