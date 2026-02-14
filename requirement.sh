pip install torch==2.3.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install \
  torch-scatter \
  torch-sparse \
  torch-cluster \
  torch-spline-conv \
  -f https://data.pyg.org/whl/torch-2.3.1+cu121.html
pip install torch-geometric
pip install pytorch-lightning tensorboard Pillow
# Optional for VG caption evaluation (not required for GQA mainline):
# pip install pycocotools git+https://github.com/salaniz/pycocoevalcap
