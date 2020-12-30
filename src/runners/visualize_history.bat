python37 ./../visualize-history.py ^
    "./../../dev-models/training_tfdata/20201229_170151 tethys_idsia_ch - regr_len63720_b64_rw_trainfrom0_ep60 - v1_model_history.pickle" ^
    "./../../dev-models/training_tfdata/20201229_221211 tethys_idsia_ch - regr_len63720_b64_rw_trainfrom0_bgCVPRindoor(len15589)_ep60 - v1_model_history.pickle" ^
    "./../../dev-models/training_tfdata/20201229_164916 tethys_idsia_ch - regr_len63720_b64_rw_trainfrom0_bgCVPRindoor(len15589,smooth)_augm095(noise)_ep60 - v1_model_history.pickle" ^
--regression ^
--loss_scale 0 0.7 ^
--r2_scale 0.4 1 ^
--acc_scale 0 1 ^
--dpi 500 ^
--save ^
--save_folder "./../../dev-models/training_tfdata/" ^