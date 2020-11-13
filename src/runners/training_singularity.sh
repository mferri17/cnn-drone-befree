# cd /home/marcofe/thesis/cnn-drone-befree/src/singularity
# singularity build --fakeroot test3.sif test3.def
# cd ..
# singularity exec --nv --bind /home/marcofe/maia/tmp:/project singularity/test3.sif runners/training_singularity.sh

python3 ./training.py \
/project/datasets/orig_train_63720 \
3 \
--regression \
--batch_size 64 \
--epochs 3 \
--retrain_from 24 \
--weights_path /project/models/v1_original_weights.pickle \
--bgs_folder /project/backgrounds/indoorCVPR_09_PPDario \
--bgs_name bg_test \
--save_folder /project/save \
--debug