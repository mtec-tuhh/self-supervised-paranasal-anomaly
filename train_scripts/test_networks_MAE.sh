

#Grad-CAM enabled testing 
CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_65_std_factor_1_Nature_DAE.yaml model=maskedAE_3D.yaml \
+seed=100 trainer.max_epochs=100 total_folds=1 current_fold=1 \
+enable_logger=False hydra.job.chdir=False datamodule.cfg.batch_size=1 \
datamodule.cfg.samples_per_patient=15 name=Masked_AE_3D_cc_65_latest hydra.job.name=Masked_AE_3D_cc_65_latest \
+load_checkpoint=Masked_AE_3D_cc_65_latest +onlyEval=False \
test_after_training=True \

