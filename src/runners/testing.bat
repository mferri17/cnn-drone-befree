python37 ./../testing.py 0 ^
--model_paths ^
    "./../../dev-models/training_tfdata/20201217_133900 tethys_idsia_ch - regr_len63720_b64_rw_trainfrom0_ep60 - v1_model.h5" ^
    "./../../dev-models/training_tfdata/20201214_222219 tethys_idsia_ch - regr_len63720_b64_rw_trainfrom0_bgCVPRindoor(len15589,smooth)_augm095(noise)_ep60 - v1_model.h5" ^
--data_folders ^
    "C:/Users/96mar/Desktop/meeting_dario/data/orig_test_11030/" ^
    "C:/Users/96mar/Desktop/meeting_dario/data/orig_train_63720/" ^
--batch_size 64 ^
--bgs_folders ^
    "C:/Users/96mar/Desktop/meeting_dario/data/aug/backgrounds_dario/indoor1/" ^
    "C:/Users/96mar/Desktop/meeting_dario/data/aug/backgrounds_dario/indoor2/" ^
--bg_smoothmask ^
--save_folder "./../../dev-models/training_tfdata/" ^