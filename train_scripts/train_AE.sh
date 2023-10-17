


CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_65_std_factor_1_Nature_DAE.yaml  model=ae_3D.yaml  \
+seed=100 trainer.max_epochs=100 total_folds=1  \
trainer.limit_train_batches=1.0  \
+enable_logger=False  \
hydra.job.chdir=False  \
datamodule.cfg.batch_size=16  \
datamodule.cfg.augmentations=False  \
datamodule.cfg.samples_per_patient=15  \
model.cfg.latent_size=128 \
name=AE_3D_cc_65_ls_128 hydra.job.name=AE_3D_cc_65_ls_128  \
+onlyEval=False test_after_training=True 


CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_65_std_factor_1_Nature_DAE.yaml  model=ae_3D.yaml  \
+seed=100 trainer.max_epochs=100 total_folds=1  \
trainer.limit_train_batches=1.0  \
+enable_logger=False  \
hydra.job.chdir=False  \
datamodule.cfg.batch_size=16  \
datamodule.cfg.augmentations=False  \
datamodule.cfg.samples_per_patient=15  \
model.cfg.latent_size=256  \
name=AE_3D_cc_65_ls_256 hydra.job.name=AE_3D_cc_65_ls_256  \
+onlyEval=False test_after_training=True 


CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_65_std_factor_1_Nature_DAE.yaml  model=ae_3D.yaml  \
+seed=100 trainer.max_epochs=100 total_folds=1  \
trainer.limit_train_batches=1.0  \
+enable_logger=False  \
hydra.job.chdir=False  \
datamodule.cfg.batch_size=16  \
datamodule.cfg.augmentations=False  \
datamodule.cfg.samples_per_patient=15  \
model.cfg.latent_size=512  \
name=AE_3D_cc_65_ls_512 hydra.job.name=AE_3D_cc_65_ls_512  \
+onlyEval=False test_after_training=True 


CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_65_std_factor_1_Nature_DAE.yaml  model=ae_3D.yaml  \
+seed=100 trainer.max_epochs=100 total_folds=1  \
trainer.limit_train_batches=1.0  \
+enable_logger=False  \
hydra.job.chdir=False  \
datamodule.cfg.batch_size=16  \
datamodule.cfg.augmentations=False  \
datamodule.cfg.samples_per_patient=15  \
model.cfg.latent_size=1024  \
name=AE_3D_cc_65_ls_1024 hydra.job.name=AE_3D_cc_65_ls_1024  \
+onlyEval=False test_after_training=True 



CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_65_std_factor_1_Nature_DAE.yaml  model=ae_3D.yaml  \
+seed=100 trainer.max_epochs=100 total_folds=1  \
trainer.limit_train_batches=1.0  \
+enable_logger=False  \
hydra.job.chdir=False  \
datamodule.cfg.batch_size=16  \
datamodule.cfg.augmentations=False  \
datamodule.cfg.samples_per_patient=15  \
model.cfg.spatial_ae=True  \
name=Spatial_AE_3D_cc_65 hydra.job.name=Spatial_AE_3D_cc_65  \
+onlyEval=False test_after_training=True 

