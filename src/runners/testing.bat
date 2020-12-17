python37 ./../testing.py 0 ^
--model_paths ^
    "./../../dev-models/training_tfdata/20201217_133900 tethys_idsia_ch - regr_len63720_b64_rw_trainfrom0_ep60 - v1_model.h5" ^
    "./../../dev-models/training_tfdata/20201214_193409 tethys_idsia_ch - regr_len63720_b64_rw_trainfrom0_bgCVPRindoor(len15589)_ep60 - v1_model.h5" ^
    "./../../dev-models/training_tfdata/20201214_184152 tethys_idsia_ch - regr_len63720_b64_rw_trainfrom0_bgCVPRindoor(len15589)_augm095_ep60 - v1_model.h5" ^
    "./../../dev-models/training_tfdata/20201214_184019 tethys_idsia_ch - regr_len63720_b64_rw_trainfrom0_bgCVPRindoor(len15589)_augm095(noise)_ep60 - v1_model.h5" ^
    "./../../dev-models/training_tfdata/20201214_222219 tethys_idsia_ch - regr_len63720_b64_rw_trainfrom0_bgCVPRindoor(len15589,smooth)_augm095(noise)_ep60 - v1_model.h5" ^
--data_folders ^
    C:/Users/96mar/Desktop/meeting_dario/data/orig_test_11030/ ^
--batch_size 64 ^
--bgs_folders ^
    None ^
    C:/Users/96mar/Desktop/meeting_dario/data/aug/backgrounds_dario/train ^
    C:/Users/96mar/Desktop/meeting_dario/data/aug/backgrounds_dario/test ^
--bgs_names ^
    bg_dario_test ^
--bg_smoothmask ^