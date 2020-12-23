

CUDA_VISIBLE_DEVICES=3 python3 ./testing.py 0 \
--model_paths \
    "/project/save/20201217_133900 tethys_idsia_ch - regr_len63720_b64_rw_trainfrom0_ep60 - v1_model.h5" \
    "/project/save/20201214_193409 tethys_idsia_ch - regr_len63720_b64_rw_trainfrom0_bgCVPRindoor(len15589)_ep60 - v1_model.h5" \
    "/project/save/20201222_164922 tethys_idsia_ch - regr_len63720_b64_rw_trainfrom0_bgCVPRindoor(len15589,smooth)_ep60 - v1_model.h5" \
    "/project/save/20201214_184152 tethys_idsia_ch - regr_len63720_b64_rw_trainfrom0_bgCVPRindoor(len15589)_augm095_ep60 - v1_model.h5" \
    "/project/save/20201214_184019 tethys_idsia_ch - regr_len63720_b64_rw_trainfrom0_bgCVPRindoor(len15589)_augm095(noise)_ep60 - v1_model.h5" \
    "/project/save/20201214_222219 tethys_idsia_ch - regr_len63720_b64_rw_trainfrom0_bgCVPRindoor(len15589,smooth)_augm095(noise)_ep60 - v1_model.h5" \
--data_folders \
    "/project/datasets/orig_test_11030/" \
    "/project/datasets/orig_train_63720/" \
--batch_size 64 \
--bgs_folders \
    "/project/backgrounds/indoor1/" \
    "/project/backgrounds/indoor2/" \
--bg_smoothmask \
--save \
--save_folder "/project/save/" \