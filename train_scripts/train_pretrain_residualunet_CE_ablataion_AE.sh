


CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_65_std_factor_1_Nature_Residual_pretraining.yaml  model=my_resUnet_3D.yaml  \
+seed=100 trainer.max_epochs=500 total_folds=1  \
trainer.limit_train_batches=1.0 model.cfg.classify=False  model.cfg.loss=ce  \
+enable_logger=False  \
hydra.job.chdir=False  \
datamodule.cfg.batch_size=128  \
datamodule.cfg.median_filter=MF_5  \
datamodule.cfg.ae_model=AE_3D_cc_65_ls_512_20  \
datamodule.cfg.samples_per_patient=1  \
name=pretrain_LARS_AE_3D_20 hydra.job.name=pretrain_LARS_AE_3D_20  \
+onlyEval=False test_after_training=True 



CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_65_std_factor_1_Nature_Residual_pretraining.yaml  model=my_resUnet_3D.yaml  \
+seed=100 trainer.max_epochs=500 total_folds=1  \
trainer.limit_train_batches=1.0 model.cfg.classify=False  model.cfg.loss=ce  \
+enable_logger=False  \
hydra.job.chdir=False  \
datamodule.cfg.batch_size=128  \
datamodule.cfg.median_filter=MF_5  \
datamodule.cfg.ae_model=AE_3D_cc_65_ls_512_40  \
datamodule.cfg.samples_per_patient=1  \
name=pretrain_LARS_AE_3D_40 hydra.job.name=pretrain_LARS_AE_3D_40  \
+onlyEval=False test_after_training=True 


CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_65_std_factor_1_Nature_Residual_pretraining.yaml  model=my_resUnet_3D.yaml  \
+seed=100 trainer.max_epochs=500 total_folds=1  \
trainer.limit_train_batches=1.0 model.cfg.classify=False  model.cfg.loss=ce  \
+enable_logger=False  \
hydra.job.chdir=False  \
datamodule.cfg.batch_size=128  \
datamodule.cfg.median_filter=MF_5  \
datamodule.cfg.ae_model=AE_3D_cc_65_ls_512_60  \
datamodule.cfg.samples_per_patient=1  \
name=pretrain_LARS_AE_3D_60 hydra.job.name=pretrain_LARS_AE_3D_60  \
+onlyEval=False test_after_training=True 


CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_65_std_factor_1_Nature_Residual_pretraining.yaml  model=my_resUnet_3D.yaml  \
+seed=100 trainer.max_epochs=500 total_folds=1  \
trainer.limit_train_batches=1.0 model.cfg.classify=False  model.cfg.loss=ce  \
+enable_logger=False  \
hydra.job.chdir=False  \
datamodule.cfg.batch_size=128  \
datamodule.cfg.median_filter=MF_5  \
datamodule.cfg.ae_model=AE_3D_cc_65_ls_512_80  \
datamodule.cfg.samples_per_patient=1  \
name=pretrain_LARS_AE_3D_80 hydra.job.name=pretrain_LARS_AE_3D_80  \
+onlyEval=False test_after_training=True 


