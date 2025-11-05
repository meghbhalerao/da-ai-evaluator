python main.py \
algorithm=diff_trans \
model=transformer \
algorithm.load_pretrained_policy=false \
algorithm.root_data_folder=/home/mb230/projects/da-ai-evaluator/da_ai_evaluator/saved_stuff/diff_trans/results/raw_results/ \
algorithm.phase=train_on_sampled \
algorithm.train_params.data_phase='interaction' \
algorithm.add_language_condition_for_interaction=false \
algorithm.train_params.batch_size=1 \
algorithm.add_language_condition=false
