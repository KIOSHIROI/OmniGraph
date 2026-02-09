# Paper-sprint profile (VG-only, stronger GQA-like transfer):
# - larger val split for more stable early-stop signal
# - less aggressive early stop (patience up, min_delta down)
# - stronger synthetic graph-QA mixing
# - single RTX 4090 optimized: fp16 + moderate batch size
python omnigraph/train/train_projector.py \
  --scene_graphs data/vg/contents/sceneGraphs/scene_graphs.json \
  --regions data/vg/contents/regionDescriptions/region_descriptions.json \
  --stage1_qformer_ckpt graph_qformer_stage1.pt \
  --graph_qa_max_per_image 5 \
  --graph_qa_repeat 3 \
  --gpu 0 \
  --batch_size 3 \
  --precision 16 \
  --max_length 256 \
  --num_workers 8 \
  --val_ratio 0.02 \
  --patience 16 \
  --min_delta 0.0005 \
  --lr 3e-5 \
  --max_steps 120000 \
  --val_check_interval 1000 \
  --save_dir checkpoints_projector_vg/stage2A_paper
