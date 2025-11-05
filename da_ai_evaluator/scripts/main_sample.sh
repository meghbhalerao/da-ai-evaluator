python main.py \
algorithm='diff_trans' \
algorithm.phase=sample \
algorithm.data_root_folder="/home/mb230/projects/hoifhli_release/data/processed_data" \
algorithm.project="./saved_stuff" \
algorithm.use_long_planned_path=true \
algorithm.add_interaction_root_xy_ori=true \
algorithm.add_interaction_feet_contact=true \
algorithm.use_guidance_in_denoising=true \
'algorithm.test_object_names=['smallbox']' \
algorithm.vis_wdir="smallbox" \
algorithm.action_name="lift" \
algorithm.finger_use_wandb=true \
algorithm.save_raw_results=true \
# --add_finger_motion \
# --vis_waypoints \
