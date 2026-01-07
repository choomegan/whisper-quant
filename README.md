# Overview
This repository quantises Whisper's encoder into TFLite format, with INT8 weights and Float32 compute.

# Quick Start
To convert Whisper model from huggingface to TFLite format

Change config at the top of the script [convert.py](convert.py).

Config should look like this:
```
MODEL_NAME = "jensenlwt/whisper-small-singlish-122k"
PT_EXPORT_DIR = "./pt_whisper_export"
TFLITE_PATH = "singlish_whisper_encoder_int8_fp32.tflite" # if you want to change tflite model name
```

Run python script to convert Whisper's encoder to TFLite
```
python3 convert.py
```

You should see a TFLite model generated in the root directory, with name specified under config.