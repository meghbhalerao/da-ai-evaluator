python main.py \
algorithm=diff_trans \
algorithm.load_pretrained_policy=true \
algorithm.phase=scene_conditioned_sample \
algorithm.use_guidance_in_denoising=true \
algorithm.add_language_condition_for_interaction=false \
algorithm.add_waypoints_xy_interaction=true \
algorithm.data_root_folder=/home/mb230/projects/hoifhli_release/data/processed_data \
algorithm.scene_type=trimesh \
