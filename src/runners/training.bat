python37 ./../training.py ^
C:/Users/96mar/Desktop/meeting_dario/data/orig_train_63720/ ^
0 ^
--regression ^
--data_len 8192 ^
--batch_size 256 ^
--epochs 1 ^
--weights_path ./../../dev-models/_originals/v1_original_weights.pickle ^
--retrain_from 0 ^
--verbose 2 ^
--lr_reducer ^
--profiler ^
--profiler_dir C:\Temp\venv\logs\tfdata-opt ^
--bgs_folder C:/Users/96mar/Desktop/meeting_dario/data/indoorCVPR_09_PPDario_uint8 ^
--bgs_len 100 ^
--bgs_name indoorCVPR_PPDario ^
--aug_prob 1 ^
--save_folder ./../../dev-models/training_generator ^
--debug