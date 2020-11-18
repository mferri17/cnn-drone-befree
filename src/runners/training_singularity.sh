# cd /home/marcofe/thesis/cnn-drone-befree/src/singularity
# singularity build --fakeroot tf-latest.sif tf-latest.def
# cd ..
# singularity exec --nv --bind /home/marcofe/thesis/data:/project singularity/tf-latest.sif runners/training_singularity.sh

CUDA_VISIBLE_DEVICES=1 python3 ./training.py \
/project/datasets/orig_train_63720 \
0 \
--regression \
--data_size 8192 \
--batch_size 256 \
--epochs 2 \
--retrain_from 0 \
--verbose 1 \
--lr_reducer \
--profiler_dir /project/logs \
--augmentation \
--bgs_folder /project/backgrounds/indoorCVPR_09_PPDario_uint8 \
--bgs_name bg_test \
--save_folder /project/save \
--debug