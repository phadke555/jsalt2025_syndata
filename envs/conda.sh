if [ -z "$1" ]; then
  echo "Error: please supply the full path for the new conda env."
  echo "Usage: $0 /path/to/env"
  exit 1
fi

ENV_PATH="/home/rphadke1/jsalt2025_syndata/envs/conda_env"

# Load Anaconda module (adjust the module name if necessary)
source ~/.bashrc

# Create a Conda environment with Python 3.10 at the specified location
echo "→ Creating conda env at ${ENV_PATH} with Python 3.10…"
conda create --prefix "${ENV_PATH}" python=3.10.15 pip -y

# Activate the Conda environment
echo "→ Activating ${ENV_PATH}…"
conda activate "${ENV_PATH}"

echo "→ Installing FFmpeg (for audio processing)…"
conda install -y -c conda-forge ffmpeg

pip install datasets[audio] soundfile pyyaml lhotse peft
pip install 'jiwer<4.0'

echo "→ Cloning F5-TTS repository…"
git clone https://github.com/SWivid/F5-TTS.git
cd F5-TTS

echo "→ Installing Python dependencies…"
pip install --upgrade pip
pip install -r requirements.txt

# Optional: install in editable mode if you plan to tweak the code
pip install -e .

# Return to project root
cd ..


echo "✅ Setup complete!"
echo "To use this environment later, run:"
echo "  conda activate ${ENV_PATH}"