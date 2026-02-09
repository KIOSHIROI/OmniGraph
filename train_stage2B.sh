# Paper-sprint profile (VG-only, stronger GQA-like transfer):
# - continue from Stage2A paper checkpoint
# - raise graph-QA density
# - relax early stopping to avoid premature convergence
# - single RTX 4090 optimized: fp16 + moderate batch size
python omnigraph/train/train_stage2B.py \
  --scene_graphs data/vg/contents/sceneGraphs/scene_graphs.json \
  --regions data/vg/contents/regionDescriptions/region_descriptions.json \
  --stage2A_ckpt checkpoints_projector_vg/stage2A_paper/last.ckpt \
  --graph_qa_max_per_image 6 \
  --graph_qa_repeat 4 \
  --gpu 0 \
  --batch_size 3 \
  --precision 16 \
  --max_length 256 \
  --lr 1.2e-5 \
  --max_steps 60000 \
  --val_ratio 0.02 \
  --val_check_interval 1000 \
  --patience 16 --min_delta 0.0005 \
  --save_dir checkpoints_projector_vg/stage2B_paper
