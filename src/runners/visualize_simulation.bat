python37 ./../visualize-simulation.py 0 ^
--models_paths ^
    "./../../dev-models/training_tfdata/20201229_170151 tethys_idsia_ch - regr_len63720_b64_rw_trainfrom0_ep60 - v1_model.h5" ^
    "./../../dev-models/training_tfdata/20201229_221211 tethys_idsia_ch - regr_len63720_b64_rw_trainfrom0_bgCVPRindoor(len15589)_ep60 - v1_model.h5" ^
    "./../../dev-models/training_tfdata/20201229_164916 tethys_idsia_ch - regr_len63720_b64_rw_trainfrom0_bgCVPRindoor(len15589,smooth)_augm095(noise)_ep60 - v1_model.h5" ^
--data_folders ^
    "C:/Users/96mar/Desktop/meeting_dario/data/custom/20210106_040707 marco_kitchen_01/" ^
    "C:/Users/96mar/Desktop/meeting_dario/data/custom/20210106_050426 marco_living_01/" ^
    "C:/Users/96mar/Desktop/meeting_dario/data/custom/20210106_050727 marco_bedroom_01/" ^
    "C:/Users/96mar/Desktop/meeting_dario/data/custom/20210106_053916 marco_bedroom_02/" ^
    "C:/Users/96mar/Desktop/meeting_dario/data/custom/20210106_185930 marco_park_01/" ^
    "C:/Users/96mar/Desktop/meeting_dario/data/custom/20210106_191242 marco_park_02/" ^
    "C:/Users/96mar/Desktop/meeting_dario/data/custom/20210106_191600 marco_courtyard_01/" ^
    "C:/Users/96mar/Desktop/meeting_dario/data/custom/20210106_191933 marco_courtyard_02/" ^
--window_sec 3 ^
--fps 30 ^
--legend_path "./../resources/legend.png" ^
--save_folder "./../../dev-visualization/flight-simulation/" ^
--debug