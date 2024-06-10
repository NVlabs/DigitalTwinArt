ROOT=$(pwd)
conda create -n artnerf python=3.8
source activate artnerf

pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install pyyaml opencv-python imageio tqdm kornia yacs einops ruamel.yaml open3d wandb

cd ${ROOT}/mycuda && rm -rf build *egg* && pip install -e .
pip install git+https://github.com/marian42/mesh_to_sdf.git
pip install kaolin==0.14.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-1.11.0_cu113.html
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
