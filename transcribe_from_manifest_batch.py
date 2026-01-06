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
BATCH_SIZE = 8


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
# Load processor + decoder
# -------------------------------------------------
processor = WhisperProcessor.from_pretrained(model_name)
decoder_model = TFWhisperForConditionalGeneration.from_pretrained(
    model_name,
    from_pt=True,
)

# -------------------------------------------------
# Load TFLite encoder
# -------------------------------------------------
interpreter = tf.lite.Interpreter(model_path=tf_model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"TFLite input shape: {input_details[0]['shape']}")

# -------------------------------------------------
# Read all manifest lines
# -------------------------------------------------
with open(manifest_path, "r") as f:
    lines = f.readlines()

num_lines = len(lines)

# -------------------------------------------------
# Process in batches
# -------------------------------------------------
with open(output_filepath, "w") as fout:
    for batch_start in tqdm(range(0, num_lines, BATCH_SIZE), desc="Processing batches"):
        batch_lines = lines[batch_start : batch_start + BATCH_SIZE]

        batch_audio_paths = []
        batch_gt_texts = []
        batch_audios = []

        for line in batch_lines:
            item = json.loads(line)
            audio_path = os.path.join(
                os.path.dirname(manifest_path), item["audio_filepath"]
            )
            batch_audio_paths.append(audio_path)
            batch_gt_texts.append(item["text"])

            audio = load_audio(audio_path)
            batch_audios.append(audio)

        # -------- Batch: Audio → log-Mel --------
        inputs = processor(
            batch_audios,
            sampling_rate=16000,
            return_tensors="tf",
            padding=True,
        )
        input_features = inputs.input_features  # (B, 80, T)

        # -------- FIX: Pad or truncate to exactly 3000 frames --------
        batch_size, n_mels, n_frames = input_features.shape

        if n_frames < 3000:
            # Pad with zeros on the right
            pad_amount = 3000 - n_frames
            input_features = tf.pad(
                input_features,
                paddings=[[0, 0], [0, 0], [0, pad_amount]],
                mode="CONSTANT",
                constant_values=0,
            )
        elif n_frames > 3000:
            # Truncate to 3000
            input_features = input_features[:, :, :3000]

        # Now shape is guaranteed to be (B, 80, 3000)

        # -------- Batch: TFLite encoder --------
        batch_encoder_hidden_states = []

        for i in range(len(batch_audios)):
            single_input = input_features[i : i + 1]  # (1, 80, 3000)

            interpreter.set_tensor(
                input_details[0]["index"],
                single_input.numpy(),
            )
            interpreter.invoke()

            encoder_hidden_states = interpreter.get_tensor(output_details[0]["index"])
            batch_encoder_hidden_states.append(encoder_hidden_states)

        # Stack encoder outputs
        batch_encoder_hidden_states = np.concatenate(
            batch_encoder_hidden_states, axis=0
        )

        # -------- Batch: TF decoder --------
        encoder_outputs = TFBaseModelOutput(
            last_hidden_state=tf.convert_to_tensor(batch_encoder_hidden_states)
        )

        generated_ids = decoder_model.generate(
            encoder_outputs=encoder_outputs,
            num_beams=1,
            max_new_tokens=128,
            do_sample=False,
        )

        batch_texts = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )

        # -------- Write batch results --------
        for audio_path, gt_text, pred_text in zip(
            batch_audio_paths, batch_gt_texts, batch_texts
        ):
            out_item = {
                "audio_filepath": audio_path,
                "text": gt_text,
                "pred_text": pred_text,
            }
            fout.write(json.dumps(out_item, ensure_ascii=False) + "\n")
            fout.flush()

print(f"✅ Transcription complete: {output_filepath}")
