CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_65_std_factor_1_Nature_DAE.yaml  model=maskedAE_3D_tiny.yaml  \
+seed=100 trainer.max_epochs=1000 total_folds=5  \
trainer.limit_train_batches=1.0 \
+enable_logger=False  \
hydra.job.chdir=False  \
datamodule.cfg.batch_size=16  \
datamodule.cfg.augmentations=False  \
datamodule.cfg.samples_per_patient=15  \
name=Masked_AE_3D_cc_65 hydra.job.name=Masked_AE_3D_cc_65  \
+onlyEval=False test_after_training=True 



CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_65_std_factor_1_Nature_DAE.yaml  model=maskedAE_3D.yaml  \
+seed=100 trainer.max_epochs=1000 total_folds=1  \
trainer.limit_train_batches=1.0 \
+enable_logger=True  \
hydra.job.chdir=False  \
datamodule.cfg.batch_size=16  \
datamodule.cfg.augmentations=False  \
datamodule.cfg.samples_per_patient=15  \
name=Masked_AE_3D_cc_65 hydra.job.name=Masked_AE_3D_cc_65  \
+onlyEval=False test_after_training=True 

