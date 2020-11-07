# singularity exec --nv --bind $HOME/Desktop/hgfs/thesis:/project tethys/test2.sif ./training_singularity.sh 

python3 ./../training.py \
/project/dataset \
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