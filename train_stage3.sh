python omnigraph/train/train_stage3.py \
  --scene_graphs data/vg/contents/sceneGraphs/scene_graphs.json \
  --regions data/vg/contents/regionDescriptions/region_descriptions.json \
  --image_root data/vg/VG_100K \
  --init_ckpt checkpoints_projector_vg/stage2A/stage2A-step=0005000-val_loss=2.607.ckpt \
  --gpu 0 --batch_size 2 --precision 32 --max_length 256 \
  --lr 5e-5 --max_steps 20000 \
  --val_ratio 0.001 --val_check_interval 2000 \
  --patience 2 --min_delta 0.01