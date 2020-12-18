CUDA_VISIBLE_DEVICES=0 python3 ./../training.py 0 ^
/project/datasets/orig_train_63720 \
--regression \
--weights_path /project/models/v1_original_weights.pickle ^
--verbose 2 \
--lr_reducer \
--debug
