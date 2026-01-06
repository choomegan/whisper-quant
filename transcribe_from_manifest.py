import json
import os
import tensorflow as tf
import soundfile as sf
import librosa
import numpy as np
from tqdm import tqdm
from transformers import WhisperProcessor, TFWhisperForConditionalGeneration
from transformers.modeling_tf_outputs import TFBaseModelOutput

print(tf.config.list_physical_devices())

# -------------------------------------------------
# Paths
# -------------------------------------------------
manifest_path = "/mnt/d/MEGAN/data/benchmark-datasets/nsc-spontaneous/manifest.json"
tf_model_path = "singlish_whisper_encoder_int8_fp32.tflite"
model_name = "openai/whisper-small"
output_filepath = "nsc_spontaneous_pred.json"


# -------------------------------------------------
# Audio loading
# -------------------------------------------------
def load_audio(path, target_sr=16000):
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio


# -------------------------------------------------
# Load processor + decoder (ONCE)
# -------------------------------------------------
processor = WhisperProcessor.from_pretrained(model_name)

decoder_model = TFWhisperForConditionalGeneration.from_pretrained(
    model_name,
    from_pt=True,
)

# -------------------------------------------------
# Load TFLite encoder (ONCE)
# -------------------------------------------------
interpreter = tf.lite.Interpreter(model_path=tf_model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# -------------------------------------------------
# Read manifest + transcribe
# -------------------------------------------------
# -------------------------------------------------
# Count lines for tqdm
# -------------------------------------------------
with open(manifest_path, "r") as f:
    num_lines = sum(1 for _ in f)

with open(output_filepath, "a") as fout:

    with open(manifest_path, "r") as f:
        for line in tqdm(f, total=num_lines, desc="Transcribing"):
            item = json.loads(line)
            audio_path = os.path.join(
                os.path.dirname(manifest_path), item["audio_filepath"]
            )
            gt_text = item["text"]

            # -------- Load audio --------
            audio = load_audio(audio_path)

            # -------- Audio → log-Mel --------
            inputs = processor(
                audio,
                sampling_rate=16000,
                return_tensors="tf",
            )
            input_features = inputs.input_features  # (1, 80, 3000)

            # -------- TFLite encoder --------
            interpreter.set_tensor(
                input_details[0]["index"],
                input_features.numpy(),
            )
            interpreter.invoke()

            encoder_hidden_states = interpreter.get_tensor(
                output_details[0]["index"]
            )  # (1, T, D)

            # -------- TF decoder --------
            encoder_outputs = TFBaseModelOutput(
                last_hidden_state=tf.convert_to_tensor(encoder_hidden_states)
            )

            generated_ids = decoder_model.generate(
                encoder_outputs=encoder_outputs,
                num_beams=1,
                max_new_tokens=128,
                do_sample=False,
            )

            text = processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
            )[0]

            out_item = {
                "audio_filepath": audio_path,
                "text": gt_text,
                "pred_text": text,
            }

            fout.write(json.dumps(out_item, ensure_ascii=False) + "\n")
            fout.flush()
