python37 ./../training.py 0 ^
C:/Users/96mar/Desktop/meeting_dario/data/orig_train_63720/ ^
--regression ^
--data_len 8192 ^
--batch_size 64 ^
--epochs 5 ^
--oversampling 3 ^
--weights_path ./../../dev-models/_originals/v1_original_weights.pickle ^
--retrain_from 0 ^
--verbose 2 ^
--lr_reducer ^
--profiler_dir C:\Temp\venv\logs\tfdata-paug ^
--bgs_folder C:/Users/96mar/Desktop/meeting_dario/data/aug/indoorCVPR_09_PPDario_uint8 ^
--bgs_len 1000 ^
--bgs_name indoorCVPR_PPDario ^
--bg_smoothmask ^
--aug_prob 0.95 ^
--noise_folder C:/Users/96mar/Desktop/meeting_dario/data/aug/perlin-noise ^
--view_stats ^
--save_folder ./../../dev-models/training_tfdata ^
--debug