# cd /home/marcofe/thesis/cnn-drone-befree/src/singularity
# singularity build --fakeroot test-tf-latest.sif test-tf-latest.def
# cd ..
# singularity exec --nv --bind /home/marcofe/thesis/data:/project singularity/test-tf-latest.sif runners/training_singularity.sh

CUDA_VISIBLE_DEVICES=0 python3 ./training.py \
/project/datasets/orig_train_63720 \
0 \
--regression \
--batch_size 64 \
--epochs 60 \
--oversampling 3 \
--retrain_from 0 \
--verbose 2 \
--lr_reducer \
--bgs_folder /project/backgrounds/indoorCVPR_09_PPDario_uint8 \
--bgs_name bgCVPRindoor \
--aug_prob 0.95 \
--noise_folder /project/perlin-noise \
--save \
--save_folder /project/save \
