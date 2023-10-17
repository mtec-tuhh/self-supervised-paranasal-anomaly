

CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_65_std_factor_1_Nature_DAE.yaml  model=ae_3D.yaml  \
+seed=100 trainer.max_epochs=100 total_folds=1  \
trainer.limit_train_batches=0.2  \
+enable_logger=False  \
hydra.job.chdir=False  \
datamodule.cfg.batch_size=16  \
datamodule.cfg.augmentations=False  \
datamodule.cfg.samples_per_patient=15  \
model.cfg.latent_size=512  \
name=AE_3D_cc_65_ls_512_20 hydra.job.name=AE_3D_cc_65_ls_512_20  \
+onlyEval=False test_after_training=True 



CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_65_std_factor_1_Nature_DAE.yaml  model=ae_3D.yaml  \
+seed=100 trainer.max_epochs=100 total_folds=1  \
trainer.limit_train_batches=0.4  \
+enable_logger=False  \
hydra.job.chdir=False  \
datamodule.cfg.batch_size=16  \
datamodule.cfg.augmentations=False  \
datamodule.cfg.samples_per_patient=15  \
model.cfg.latent_size=512  \
name=AE_3D_cc_65_ls_512_40 hydra.job.name=AE_3D_cc_65_ls_512_40  \
+onlyEval=False test_after_training=True




CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_65_std_factor_1_Nature_DAE.yaml  model=ae_3D.yaml  \
+seed=100 trainer.max_epochs=100 total_folds=1  \
trainer.limit_train_batches=0.6  \
+enable_logger=False  \
hydra.job.chdir=False  \
datamodule.cfg.batch_size=16  \
datamodule.cfg.augmentations=False  \
datamodule.cfg.samples_per_patient=15  \
model.cfg.latent_size=512  \
name=AE_3D_cc_65_ls_512_60 hydra.job.name=AE_3D_cc_65_ls_512_60  \
+onlyEval=False test_after_training=True


CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_65_std_factor_1_Nature_DAE.yaml  model=ae_3D.yaml  \
+seed=100 trainer.max_epochs=100 total_folds=1  \
trainer.limit_train_batches=0.8  \
+enable_logger=False  \
hydra.job.chdir=False  \
datamodule.cfg.batch_size=16  \
datamodule.cfg.augmentations=False  \
datamodule.cfg.samples_per_patient=15  \
model.cfg.latent_size=512  \
name=AE_3D_cc_65_ls_512_80 hydra.job.name=AE_3D_cc_65_ls_512_80  \
+onlyEval=False test_after_training=True

