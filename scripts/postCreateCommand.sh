mamba env update --file scripts/postcreaterequirements.yml

pip install cupy-cuda11x tensorboardX hydra-core --upgrade

pip3 install torch torchvision torchaudio

cd ~/

git clone https://github.com/gschramm/pyparallelproj.git

echo "export PYTHONPATH="${PYTHONPATH}:/home/user/pyparallelproj/"" | sudo tee -a "/home/user/.bashrc"

source "/home/user/.bashrc"