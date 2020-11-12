# cd /home/marcofe/thesis/cnn-drone-befree/src/singularity
# singularity build --fakeroot test3.sif test3.def
# cd ..
# singularity exec --nv --bind /home/marcofe/maia/tmp:/project singularity/test3.sif runners/training_singularity.sh

python3 ./training.py \
/project/dataset \
3 \
--regression \
--data_size 1024 \
--batch_size 64 \
--epochs 5 \
--weights_path /project/v1_original_weights.pickle \
--retrain_from 24 \
--augmentation \
--bgs_folder /project/backgrounds \
--bgs_name bg_test \
--debug
