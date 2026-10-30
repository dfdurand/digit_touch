#!/usr/bin/env python3
"""
digit_touch: int8 quantization of the model and export for TFLM.

Usage:
    source ~/venvs/tf/bin/activate
    python quantize.py

Input:   train/out/model.keras (produced by train.py)
Outputs:
    train/out/model_float.tflite   float32 version (to compare sizes)
    train/out/model_int8.tflite    int8 version, the one that goes on the board
    train/out/quant.json           accuracies, scale/zero_point, operations, sizes
    src/model_data.c / .h          the int8 .tflite as a C array
"""

import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import tensorflow as tf
from tensorflow import keras

TRAIN_DIR = Path(__file__).resolve().parent
OUT = TRAIN_DIR / "out"
SRC = TRAIN_DIR.parent / "src"
N_REPRESENTATIVE = 500
SEED = 42


def load_mnist():
    (x_train, _), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = (x_train.astype("float32") / 255.0)[..., np.newaxis]
    x_test = (x_test.astype("float32") / 255.0)[..., np.newaxis]
    return x_train, x_test, y_test


# --------------------------------------------------------------------------
# 1. Conversion
# --------------------------------------------------------------------------
def make_converter(model):
    """Converter with a batch size FIXED to 1.

    By default, Keras keeps a variable batch (None). The Flatten then has to
    compute its shape at runtime, which adds the SHAPE, STRIDED_SLICE and
    PACK operations. On the board, we always process 1 image: with a fixed
    shape, the Flatten becomes a simple RESHAPE.
    """
    # Same architecture, but with an input declared with batch_size=1, then
    # the trained weights are copied layer by layer.
    inp = keras.Input(shape=(28, 28, 1), batch_size=1, name="image")
    x = inp
    for layer in model.layers:
        x = layer.__class__.from_config(layer.get_config())(x)
    fixed = keras.Model(inp, x, name=model.name)
    fixed.set_weights(model.get_weights())
    return tf.lite.TFLiteConverter.from_keras_model(fixed)


def convert_float(model):
    """Plain conversion to the .tflite format, weights still in float32."""
    return make_converter(model).convert()


def convert_int8(model, x_train):
    """Full post-training int8 quantization.

    The converter runs the "representative" images through the model to
    observe the value range (min..max) of each intermediate tensor. From
    that it derives, for each one, the scale and zero_point that fit this
    range into the 256 values of an int8.
    """
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(x_train), N_REPRESENTATIVE, replace=False)

    def representative_dataset():
        for i in idx:
            yield [x_train[i:i + 1]]          # one image at a time, shape (1, 28, 28, 1)

    conv = make_converter(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]          # enables quantization
    conv.representative_dataset = representative_dataset     # to calibrate the activations
    # int8 operations only: if a layer could not be quantized, the conversion
    # would fail instead of silently keeping float.
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # The input and output themselves are int8: the board will send int8
    # directly and receive int8, without any float computation.
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    return conv.convert()


# --------------------------------------------------------------------------
# 2. Verification with the TFLite interpreter (what TFLM will do on the board)
# --------------------------------------------------------------------------
def make_interpreter(tflite_bytes):
    # Without the XNNPACK delegate (PC accelerator): the operations are run
    # as they are in the file, as TFLM will do on the board.
    interp = tf.lite.Interpreter(
        model_content=tflite_bytes,
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
    interp.allocate_tensors()
    return interp


def run_int8(interp, x):
    """Run the int8 model on all images, as the C code will."""
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    in_scale, in_zp = inp["quantization"]
    out_scale, out_zp = out["quantization"]

    probs = np.empty((len(x), 10), dtype=np.float32)
    for i in range(len(x)):
        # float [0,1] -> int8: q = round(x / scale) + zero_point, clamped to [-128, 127]
        q = np.clip(np.round(x[i:i + 1] / in_scale) + in_zp, -128, 127).astype(np.int8)
        interp.set_tensor(inp["index"], q)
        interp.invoke()
        q_out = interp.get_tensor(out["index"])[0].astype(np.float32)
        # int8 -> float: real = scale * (q - zero_point)
        probs[i] = out_scale * (q_out - out_zp)
    return probs


def run_float(interp, x):
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    probs = np.empty((len(x), 10), dtype=np.float32)
    for i in range(len(x)):
        interp.set_tensor(inp["index"], x[i:i + 1])
        interp.invoke()
        probs[i] = interp.get_tensor(out["index"])[0]
    return probs


def list_ops(interp):
    """TFLite operations contained in the model (for the MicroMutableOpResolver)."""
    return [op["op_name"] for op in interp._get_ops_details()]


def activations(interp):
    """Activation tensors (input + output of each operation), with their size.

    Weights and biases are constants: they stay in flash, in the model
    array. Activations, on the other hand, are computed at each inference
    and live in the tensor arena (RAM).
    """
    details = {t["index"]: t for t in interp.get_tensor_details()}
    idx = [interp.get_input_details()[0]["index"]]
    for op in interp._get_ops_details():
        idx += list(op["outputs"])
    res = []
    for i in idx:
        t = details[i]
        res.append({"name": t["name"], "shape": t["shape"].tolist(),
                    "bytes": int(np.prod(t["shape"])) * np.dtype(t["dtype"]).itemsize})
    return res


# --------------------------------------------------------------------------
# 3. Show concretely what quantization does to the weights
# --------------------------------------------------------------------------
def show_weight_example(model, interp):
    """Compare a 3x3 conv1 filter in float32 and in int8."""
    w_float = model.get_layer("conv1").get_weights()[0][:, :, 0, 0]   # filter #0, 3x3

    for t in interp.get_tensor_details():
        q = t["quantization_parameters"]
        shape = tuple(t["shape"])
        if t["dtype"] == np.int8 and shape == (8, 3, 3, 1):          # conv1 weights
            w_q = interp.get_tensor(t["index"])[0, :, :, 0]
            scale = float(q["scales"][0])
            zp = int(q["zero_points"][0])
            return {
                "filter_float32": np.round(w_float, 4).tolist(),
                "scale": scale,
                "zero_point": zp,
                "filter_int8": w_q.astype(int).tolist(),
                "filter_reconstructed": np.round(scale * (w_q.astype(np.float32) - zp), 4).tolist(),
                "max_error": float(np.abs(scale * (w_q - zp) - w_float).max()),
            }
    return None


# --------------------------------------------------------------------------
# 4. Export as a C array
# --------------------------------------------------------------------------
def write_c_array(tflite_bytes, info):
    hex_lines = []
    for i in range(0, len(tflite_bytes), 12):
        chunk = tflite_bytes[i:i + 12]
        hex_lines.append("\t" + ", ".join(f"0x{b:02x}" for b in chunk) + ",")

    (SRC / "model_data.h").write_text(f"""/*
 * digit_touch int8 CNN model — FILE GENERATED by train/quantize.py, do not edit.
 *
 * Input      : int8 [1][28][28][1], scale={info['input']['scale']:.8f} zero_point={info['input']['zero_point']}
 * Output     : int8 [1][10],        scale={info['output']['scale']:.8f} zero_point={info['output']['zero_point']}
 * Operations : {", ".join(info['unique_operations'])}
 */

#ifndef MODEL_DATA_H_
#define MODEL_DATA_H_

#ifdef __cplusplus
extern "C" {{
#endif

extern const unsigned char g_model_data[];
extern const unsigned int g_model_data_len;

#ifdef __cplusplus
}}
#endif

#endif /* MODEL_DATA_H_ */
""")

    (SRC / "model_data.c").write_text(
        "/* FILE GENERATED by train/quantize.py, do not edit. */\n\n"
        '#include "model_data.h"\n\n'
        "/* TFLM requires the model to be 16-byte aligned */\n"
        "const unsigned char g_model_data[] __attribute__((aligned(16))) = {\n"
        + "\n".join(hex_lines)
        + f"\n}};\n\nconst unsigned int g_model_data_len = {len(tflite_bytes)};\n"
    )


# --------------------------------------------------------------------------
# 5. Main program
# --------------------------------------------------------------------------
def main():
    model = keras.models.load_model(OUT / "model.keras")
    x_train, x_test, y_test = load_mnist()

    print("Conversion float32...")
    tfl_float = convert_float(model)
    print("Conversion int8 (calibration on", N_REPRESENTATIVE, "images)...")
    tfl_int8 = convert_int8(model, x_train)
    (OUT / "model_float.tflite").write_bytes(tfl_float)
    (OUT / "model_int8.tflite").write_bytes(tfl_int8)

    print("Evaluating the 3 versions on the 10,000 test images...")
    acc_keras = float((model.predict(x_test, verbose=0).argmax(1) == y_test).mean())
    it_float = make_interpreter(tfl_float)
    p_float = run_float(it_float, x_test)
    it_int8 = make_interpreter(tfl_int8)
    p_int8 = run_int8(it_int8, x_test)

    pred_float, pred_int8 = p_float.argmax(1), p_int8.argmax(1)
    inp = it_int8.get_input_details()[0]
    out = it_int8.get_output_details()[0]
    ops = list_ops(it_int8)
    acts = activations(it_int8)
    disagreements = np.where(pred_float != pred_int8)[0]

    info = {
        "float_size_bytes": len(tfl_float),
        "int8_size_bytes": len(tfl_int8),
        "reduction": round(len(tfl_float) / len(tfl_int8), 2),
        "keras_accuracy_pct": round(100 * acc_keras, 2),
        "tflite_float_accuracy_pct": round(100 * float((pred_float == y_test).mean()), 2),
        "tflite_int8_accuracy_pct": round(100 * float((pred_int8 == y_test).mean()), 2),
        "float_vs_int8_disagreements": int(len(disagreements)),
        "mean_prob_difference": round(float(np.abs(p_float - p_int8).mean()), 5),
        "input": {"type": str(inp["dtype"].__name__), "shape": inp["shape"].tolist(),
                  "scale": float(inp["quantization"][0]),
                  "zero_point": int(inp["quantization"][1])},
        "output": {"type": str(out["dtype"].__name__), "shape": out["shape"].tolist(),
                   "scale": float(out["quantization"][0]),
                   "zero_point": int(out["quantization"][1])},
        "operations": ops,
        "unique_operations": sorted(set(ops)),
        "activations": [f"{a['shape']}: {a['bytes']} B" for a in acts],
        "largest_activation_bytes": max(a["bytes"] for a in acts),
        # At worst, TFLM must hold the input and output of an operation at the same time
        "largest_input_output_pair_bytes": max(
            acts[i]["bytes"] + acts[i + 1]["bytes"] for i in range(len(acts) - 1)),
        "conv1_filter_example": show_weight_example(model, it_int8),
    }
    (OUT / "quant.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))
    write_c_array(tfl_int8, info)

    print("\n================ SUMMARY ================")
    print(json.dumps(info, indent=2, ensure_ascii=False))
    print(f"\nWritten: {OUT}/model_int8.tflite, {SRC}/model_data.c, {SRC}/model_data.h")


if __name__ == "__main__":
    main()
