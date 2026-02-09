# Paper-sprint profile (VG-only, stronger GQA-like transfer):
# - initialize from Stage2B paper checkpoint
# - keep region caption objective while adding graph-QA mix
# - slower lr and longer schedule for multimodal alignment
# - single RTX 4090 optimized: fp16, keep batch conservative for tri-modal stage
python omnigraph/train/train_stage3.py \
  --scene_graphs data/vg/contents/sceneGraphs/scene_graphs.json \
  --regions data/vg/contents/regionDescriptions/region_descriptions.json \
  --image_root data/vg \
  --stage2B_ckpt checkpoints_projector_vg/stage2B_paper/last.ckpt \
  --graph_qa_max_per_image 4 \
  --graph_qa_repeat 2 \
  --gpu 0 --batch_size 2 --precision 16 --max_length 256 \
  --lr 2e-5 --max_steps 50000 \
  --val_ratio 0.02 --val_check_interval 1000 \
  --patience 14 --min_delta 0.0005 \
  --save_dir checkpoints_stage3_paper
