import os
import sys

# Locate the site-packages directory in the active environment
site_packages = next(p for p in sys.path if 'site-packages' in p)

# Point Windows to the PyTorch CUDA libraries
torch_lib = os.path.join(site_packages, "torch", "lib")
if os.path.exists(torch_lib):
    os.add_dll_directory(torch_lib)

import birdnet
from pathlib import Path

def main():
# 1. Load the model into GPU
    print("Loading BirdNET V3 ONNX model...")
    model = birdnet.load("acoustic", "3.0", "onnx")

    input_folder = Path(r"C:\Users\Spycrab\Documents\Uni\ACUS220\audio")
    output_folder = Path(r"C:\Users\Spycrab\Documents\Uni\ACUS220\output")
    output_folder.mkdir(parents=True, exist_ok=True)

    for audio_file in input_folder.rglob("*.wav"):
        print(f"Analyzing {audio_file.name}...")

        predictions = model.predict(audio_file)
    
        output_file = output_folder / f"{audio_file.stem}.csv"
        predictions.to_csv(output_file)

    print("Batch processing complete!")

if __name__ == '__main__':
    main()
