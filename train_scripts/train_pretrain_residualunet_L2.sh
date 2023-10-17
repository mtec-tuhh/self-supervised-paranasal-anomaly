




CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_65_std_factor_1_Nature_Residual_pretraining.yaml  model=my_resUnet_3D.yaml  \
+seed=100 trainer.max_epochs=500 total_folds=1  \
trainer.limit_train_batches=1.0 model.cfg.classify=False  model.cfg.loss=l2  \
+enable_logger=False  \
hydra.job.chdir=False  \
datamodule.cfg.batch_size=128  \
datamodule.cfg.median_filter=MF_0  \
datamodule.cfg.samples_per_patient=1  \
name=pretrain_LARS_residual_net_bs_128_l2_MF_0 hydra.job.name=pretrain_LARS_residual_net_bs_128_l2_MF_0  \
+onlyEval=False test_after_training=True 


CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_65_std_factor_1_Nature_Residual_pretraining.yaml  model=my_resUnet_3D.yaml  \
+seed=100 trainer.max_epochs=500 total_folds=1  \
trainer.limit_train_batches=1.0 model.cfg.classify=False  model.cfg.loss=l2  \
+enable_logger=False  \
hydra.job.chdir=False  \
datamodule.cfg.batch_size=128  \
datamodule.cfg.median_filter=MF_3  \
datamodule.cfg.samples_per_patient=1  \
name=pretrain_LARS_residual_net_bs_128_l2_MF_3 hydra.job.name=pretrain_LARS_residual_net_bs_128_l2_MF_3  \
+onlyEval=False test_after_training=True 


CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_65_std_factor_1_Nature_Residual_pretraining.yaml  model=my_resUnet_3D.yaml  \
+seed=100 trainer.max_epochs=500 total_folds=1  \
trainer.limit_train_batches=1.0 model.cfg.classify=False  model.cfg.loss=l2  \
+enable_logger=False  \
hydra.job.chdir=False  \
datamodule.cfg.batch_size=128  \
datamodule.cfg.median_filter=MF_5  \
datamodule.cfg.samples_per_patient=1  \
name=pretrain_LARS_residual_net_bs_128_l2_MF_5 hydra.job.name=pretrain_LARS_residual_net_bs_128_l2_MF_5  \
+onlyEval=False test_after_training=True 



CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_65_std_factor_1_Nature_Residual_pretraining.yaml  model=my_resUnet_3D.yaml  \
+seed=100 trainer.max_epochs=500 total_folds=1  \
trainer.limit_train_batches=1.0 model.cfg.classify=False  model.cfg.loss=l2  \
+enable_logger=False  \
hydra.job.chdir=False  \
datamodule.cfg.batch_size=128  \
datamodule.cfg.median_filter=MF_7  \
datamodule.cfg.samples_per_patient=1  \
name=pretrain_LARS_residual_net_bs_128_l2_MF_7 hydra.job.name=pretrain_LARS_residual_net_bs_128_l2_MF_7  \
+onlyEval=False test_after_training=True 
