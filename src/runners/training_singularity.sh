# cd /home/marcofe/thesis/cnn-drone-befree/src/singularity
# singularity build --fakeroot test3.sif test3.def
# cd ..
# singularity exec --nv --bind /home/marcofe/maia/data:/project singularity/test3.sif runners/training_singularity.sh

CUDA_VISIBLE_DEVICES=0 python3 ./training.py \
/project/datasets/orig_train_63720 \
0 \
--regression \
--batch_size 64 \
--epochs 3 \
--weights_path /project/models/v1_original_weights.pickle \
--retrain_from 24 \
--verbose 1 \
--lr_reducer \
--bgs_folder /project/backgrounds/indoorCVPR_09_PPDario \
--bgs_name bg_test \
--save_folder /project/save \
--debug