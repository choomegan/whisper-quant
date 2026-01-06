import tensorflow as tf
import soundfile as sf
import librosa
import numpy as np
from transformers import WhisperProcessor, TFWhisperForConditionalGeneration


audio_path = "/home/pumba-04/Desktop/MEGAN/datasets/NSC_standardized/train/DIFF/WAVE/10140792.wav"


# -------------------------
# Audio loading
# -------------------------
def load_audio(path, target_sr=16000):
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio


# -------------------------
# Processor
# -------------------------
model_name = "openai/whisper-small"
processor = WhisperProcessor.from_pretrained(model_name)

audio = load_audio(audio_path)

# Audio → log-Mel (matches encoder input [1, 80, 3000])
inputs = processor(
    audio,
    sampling_rate=16000,
    return_tensors="tf",
)
input_features = inputs.input_features  # (1, 80, 3000)


# -------------------------
# TFLite Encoder Inference
# -------------------------
interpreter = tf.lite.Interpreter(
    model_path="whisper_encoder_int8_weights_fp32_compute.tflite"
)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Set input
interpreter.set_tensor(
    input_details[0]["index"],
    input_features.numpy(),
)

# Run encoder
interpreter.invoke()

# Get encoder hidden states
encoder_hidden_states = interpreter.get_tensor(
    output_details[0]["index"]
)

print("Encoder output shape:", encoder_hidden_states.shape)
# Expected: (1, time_steps, hidden_dim)


# -------------------------
# Decoder (TF)
# -------------------------
decoder_model = TFWhisperForConditionalGeneration.from_pretrained(
    model_name,
    from_pt=True,
)

generated_ids = decoder_model.generate(
    encoder_outputs=(tf.convert_to_tensor(encoder_hidden_states),),
    max_length=448,
)

# Decode tokens → text
text = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True,
)[0]

print("Transcription:")
print(text)
