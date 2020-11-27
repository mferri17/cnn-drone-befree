# cd /home/marcofe/thesis/cnn-drone-befree/src/singularity
# singularity build --fakeroot tf-latest.sif tf-latest.def
# cd ..
# singularity exec --nv --bind /home/marcofe/thesis/data:/project singularity/tf-latest.sif runners/training_singularity.sh

CUDA_VISIBLE_DEVICES=1 python3 ./training.py \
/project/datasets/orig_train_63720 \
0 \
--regression \
--batch_size 64 \
--epochs 200 \
--retrain_from 0 \
--verbose 2 \
--lr_reducer \
--bgs_folder /project/backgrounds/indoorCVPR_09_PPDario_uint8 \
--bgs_name bgCVPRindoor \
--aug_prob 0.95 \
--save \
--save_folder /project/save \
