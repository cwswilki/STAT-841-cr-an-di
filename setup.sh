#!/bin/bash

# Stop the script if any command fails
set -e 

echo "--- Creating Python Virtual Environment ---"
# This will now work correctly because we installed python3-venv
python3 -m venv venv

# 1. UPGRADE pip (Call the venv pip directly)
echo "--- Upgrading Pip ---"
./venv/bin/pip install --upgrade pip

# 2. INSTALL PyTorch (Call the venv pip directly)
echo "--- Installing PyTorch with CUDA 13 Support ---"
./venv/bin/pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

# 3. INSTALL dependencies
echo "--- Installing Additional Requirements ---"
./venv/bin/pip install -r requirements.txt

# 4. VERIFY Installation (Call the venv python directly)
echo "--- Verifying GPU Access ---"
./venv/bin/python -c "import torch; print(f'Torch Version: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'Device Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

echo -e "\nSetup Complete. Activate with: source venv/bin/activate"

