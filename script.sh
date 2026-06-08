# nvidia-htop.py -l

# cp -r ../gaussian-splatting/output/bicycle_rerender/train/ours_30000/renders/ .

# python train_wm.py -s data/360/garden --eval -m output/garden_bicycle_5w --iterations 50000 -r 4
# python render.py -m output/garden_bicycle_5w  -s data/360/garden --eval

# python metrics.py -m output/room_kitchen_attention_5w

# python train_wm.py -s data/360/room --eval -m output/room_kitchen_attention_v1p_5w --iterations 50000 -r 4
# python render.py -m output/room_kitchen_attention_v1p_5w  -s data/360/room --eval
# python metrics.py -m output/room_kitchen_attention_v1p_5w


# python train_wm.py -s data/360/kitchen --eval -m output/kitchen_room --iterations 50000 -r 4
python render.py -m output/kitchen_room --eval
# python metrics.py -m output/kitchen_room