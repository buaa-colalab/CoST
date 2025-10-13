# train v2v4real
python opencood/tools/train.py --hypes_yaml opencood/hypes_yaml/point_pillar_cost.yaml \

# train v2v4real multi cards
# python -m torch.distributed.launch --nproc_per_node=8  --master_port 29505 \
# --use_env opencood/tools/train.py --hypes_yaml opencood/hypes_yaml/point_pillar_cost.yaml

# train stt module
# python -m torch.distributed.launch --nproc_per_node=8  --master_port 29505 \
# --use_env opencood/tools/train.py --hypes_yaml opencood/hypes_yaml/point_pillar_cost_compress.yaml \
# --finetune_from checkpoint_path --freeze


