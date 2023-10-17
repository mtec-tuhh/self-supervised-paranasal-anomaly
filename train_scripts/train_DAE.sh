


CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_45_std_factor_1_Nature_DAE.yaml  model=dae_3D.yaml  \
+seed=100 trainer.max_epochs=100 total_folds=5  \
trainer.limit_train_batches=1.0  \
+enable_logger=True  \
hydra.job.chdir=False  \
datamodule.cfg.batch_size=16  \
datamodule.cfg.samples_per_patient=15  \
datamodule.cfg.noise_std=0.1  \
model.cfg.unet_model=UNET \
name=DAE_UNet3D_cc_45_noise_01 hydra.job.name=DAE_UNet3D_cc_45_noise_01  \
+onlyEval=False test_after_training=True 

CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_45_std_factor_1_Nature_DAE.yaml  model=dae_3D.yaml  \
+seed=100 trainer.max_epochs=100 total_folds=5  \
trainer.limit_train_batches=1.0  \
+enable_logger=True  \
hydra.job.chdir=False  \
datamodule.cfg.batch_size=16  \
datamodule.cfg.samples_per_patient=15  \
datamodule.cfg.noise_std=0.2  \
model.cfg.unet_model=UNET \
name=DAE_UNet3D_cc_45_noise_02 hydra.job.name=DAE_UNet3D_cc_45_noise_02  \
+onlyEval=False test_after_training=True 

CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_45_std_factor_1_Nature_DAE.yaml  model=dae_3D.yaml  \
+seed=100 trainer.max_epochs=100 total_folds=5  \
trainer.limit_train_batches=1.0  \
+enable_logger=True  \
hydra.job.chdir=False  \
datamodule.cfg.batch_size=16  \
datamodule.cfg.samples_per_patient=15  \
datamodule.cfg.noise_std=0.3  \
model.cfg.unet_model=UNET \
name=DAE_UNet3D_cc_45_noise_03 hydra.job.name=DAE_UNet3D_cc_45_noise_03  \
+onlyEval=False test_after_training=True 


CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_45_std_factor_1_Nature_DAE.yaml  model=dae_3D.yaml  \
+seed=100 trainer.max_epochs=100 total_folds=5  \
trainer.limit_train_batches=1.0  \
+enable_logger=True  \
hydra.job.chdir=False  \
datamodule.cfg.batch_size=16  \
datamodule.cfg.samples_per_patient=15  \
model.cfg.unet_model=UNET \
datamodule.cfg.noise_std=0.4  \
name=DAE_UNet3D_cc_45_noise_04 hydra.job.name=DAE_UNet3D_cc_45_noise_04  \
+onlyEval=False test_after_training=True 


CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_45_std_factor_1_Nature_DAE.yaml  model=dae_3D.yaml  \
+seed=100 trainer.max_epochs=100 total_folds=5  \
trainer.limit_train_batches=1.0  \
+enable_logger=True  \
hydra.job.chdir=False  \
datamodule.cfg.batch_size=16  \
datamodule.cfg.samples_per_patient=15  \
model.cfg.unet_model=UNET \
datamodule.cfg.noise_std=0.5  \
name=DAE_UNet3D_cc_45_noise_05 hydra.job.name=DAE_UNet3D_cc_45_noise_05  \
+onlyEval=False test_after_training=True 


CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_45_std_factor_1_Nature_DAE.yaml  model=dae_3D.yaml  \
+seed=100 trainer.max_epochs=100 total_folds=5  \
trainer.limit_train_batches=1.0  \
+enable_logger=True  \
hydra.job.chdir=False  \
datamodule.cfg.batch_size=16  \
datamodule.cfg.samples_per_patient=15  \
model.cfg.unet_model=UNET \
datamodule.cfg.noise_std=0.6  \
name=DAE_UNet3D_cc_45_noise_06 hydra.job.name=DAE_UNet3D_cc_45_noise_06  \
+onlyEval=False test_after_training=True 


CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_45_std_factor_1_Nature_DAE.yaml  model=dae_3D.yaml  \
+seed=100 trainer.max_epochs=100 total_folds=5  \
trainer.limit_train_batches=1.0  \
+enable_logger=True  \
hydra.job.chdir=False  \
datamodule.cfg.batch_size=16  \
datamodule.cfg.samples_per_patient=15  \
model.cfg.unet_model=UNET \
datamodule.cfg.noise_std=0.7  \
name=DAE_UNet3D_cc_45_noise_07 hydra.job.name=DAE_UNet3D_cc_45_noise_07  \
+onlyEval=False test_after_training=True 



CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_45_std_factor_1_Nature_DAE.yaml  model=dae_3D.yaml  \
+seed=100 trainer.max_epochs=100 total_folds=5  \
trainer.limit_train_batches=1.0  \
+enable_logger=True  \
hydra.job.chdir=False  \
datamodule.cfg.batch_size=16  \
datamodule.cfg.samples_per_patient=15  \
model.cfg.unet_model=UNET \
datamodule.cfg.noise_std=0.8  \
name=DAE_UNet3D_cc_45_noise_08 hydra.job.name=DAE_UNet3D_cc_45_noise_08  \
+onlyEval=False test_after_training=True 