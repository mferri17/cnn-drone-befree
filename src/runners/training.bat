python37 ./../training.py ^
C:/Users/96mar/Desktop/meeting_dario/data/orig_train_63720/ ^
0 ^
--regression ^
--data_size 8192 ^
--batch_size 256 ^
--epochs 2 ^
--weights_path ./../../dev-models/_originals/v1_original_weights.pickle ^
--retrain_from 24 ^
--verbose 2 ^
--lr_reducer ^
--profiler_dir C:\Temp\venv\logs\tfdata ^
--augmentation ^
--bgs_folder C:/Users/96mar/Desktop/meeting_dario/data/indoorCVPR_09_PPDario_uint8 ^
--bgs_name indoorCVPR_PPDario ^
--save_folder ./../../dev-models/training_generator ^
--debug