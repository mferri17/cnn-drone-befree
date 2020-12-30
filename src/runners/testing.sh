

CUDA_VISIBLE_DEVICES=0 python3 ./testing.py 0 \
--model_paths \
    "/project/save/20201229_170151 tethys_idsia_ch - regr_len63720_b64_rw_trainfrom0_ep60 - v1_model.h5" \
    "/project/save/20201229_221211 tethys_idsia_ch - regr_len63720_b64_rw_trainfrom0_bgCVPRindoor(len15589)_ep60 - v1_model.h5" \
    "/project/save/20201229_172245 tethys_idsia_ch - regr_len63720_b64_rw_trainfrom0_bgCVPRindoor(len15589,smooth)_ep60 - v1_model.h5" \
    "/project/save/20201229_232222 tethys_idsia_ch - regr_len63720_b64_rw_trainfrom0_bgCVPRindoor(len15589)_augm095_ep60 - v1_model.h5" \
    "/project/save/20201230_013145 tethys_idsia_ch - regr_len63720_b64_rw_trainfrom0_bgCVPRindoor(len15589)_augm095(noise)_ep60 - v1_model.h5" \
    "/project/save/20201229_164916 tethys_idsia_ch - regr_len63720_b64_rw_trainfrom0_bgCVPRindoor(len15589,smooth)_augm095(noise)_ep60 - v1_model.h5" \
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