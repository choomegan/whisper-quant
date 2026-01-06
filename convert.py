import os
import tensorflow as tf
import numpy as np

from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    TFWhisperForConditionalGeneration,
)

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
MODEL_NAME = "jensenlwt/whisper-small-singlish-122k"
PT_EXPORT_DIR = "./pt_whisper_export"
TFLITE_PATH = "singlish_whisper_encoder_int8_fp32.tflite"

# ------------------------------------------------------------
# Step 1: Load PyTorch model (supports safetensors)
# ------------------------------------------------------------
print("Loading PyTorch Whisper model (safetensors)...")
pt_model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)

# ------------------------------------------------------------
# Step 2: Save PyTorch weights as pytorch_model.bin
# (TensorFlow cannot load safetensors directly)
# ------------------------------------------------------------
print("Exporting PyTorch weights to disk...")
pt_model.save_pretrained(
    PT_EXPORT_DIR,
    safe_serialization=False,  # IMPORTANT: forces pytorch_model.bin
)

# ------------------------------------------------------------
# Step 3: Load TensorFlow model from PyTorch checkpoint
# ------------------------------------------------------------
print("Loading TensorFlow Whisper model from PyTorch weights...")
tf_model = TFWhisperForConditionalGeneration.from_pretrained(
    PT_EXPORT_DIR,
    from_pt=True,
)

# ------------------------------------------------------------
# Step 4: Extract encoder
# ------------------------------------------------------------
encoder = tf_model.model.encoder


# ------------------------------------------------------------
# Step 5: Define concrete function
# Whisper encoder expects: [B, 80, 3000]
# ------------------------------------------------------------
@tf.function(input_signature=[tf.TensorSpec(shape=[1, 80, 3000], dtype=tf.float32)])
def encoder_fn(input_features):
    # return last_hidden_state only
    return encoder(input_features, return_dict=False)[0]


concrete_fn = encoder_fn.get_concrete_function()

# ------------------------------------------------------------
# Step 6: Convert to TFLite (INT8 weights, FP32 compute)
# ------------------------------------------------------------
print("Converting encoder to TFLite (INT8 weights, FP32 compute)...")

converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_fn])

# Enable post-training weight quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# IMPORTANT:
# - Do NOT set inference_input_type / output_type
# - Do NOT force INT8 ops
# This keeps compute in FP32 and only quantizes weights
tflite_model = converter.convert()

# ------------------------------------------------------------
# Step 7: Save TFLite model
# ------------------------------------------------------------
with open(TFLITE_PATH, "wb") as f:
    f.write(tflite_model)

# ------------------------------------------------------------
# Step 8: Report size
# ------------------------------------------------------------
original_size_mb = sum(w.numpy().nbytes for w in encoder.trainable_weights) / (1024**2)
quantized_size_mb = len(tflite_model) / (1024**2)

print("\n✅ Conversion complete")
print(f"Original encoder size : {original_size_mb:.2f} MB")
print(f"TFLite encoder size   : {quantized_size_mb:.2f} MB")
print(f"Compression ratio    : {original_size_mb / quantized_size_mb:.2f}x")
print("Model type           : INT8 weights, FP32 compute")
