




CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_65_std_factor_1_Classification_Nature_10.yaml  model=my_resUnet_3D_simclr.yaml  \
+seed=100 trainer.max_epochs=100 total_folds=5  \
trainer.limit_train_batches=1.0 \
+enable_logger=False model.cfg.classify=True \
hydra.job.chdir=False  \
datamodule.cfg.batch_size=16  \
datamodule.cfg.samples_per_patient=1  \
+load_checkpoint_for_training=pretrain_simclr_t_007 \
name=finetune_new_residual_net_simclr_007_10 hydra.job.name=finetune_new_residual_net_simclr_007_10  \
+onlyEval=False test_after_training=True 



CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_65_std_factor_1_Classification_Nature_80.yaml  model=my_resUnet_3D_simclr.yaml  \
+seed=100 trainer.max_epochs=100 total_folds=5  \
trainer.limit_train_batches=1.0 \
+enable_logger=False model.cfg.classify=True \
hydra.job.chdir=False  \
datamodule.cfg.batch_size=16  \
datamodule.cfg.samples_per_patient=1  \
+load_checkpoint_for_training=pretrain_simclr_t_007 \
name=finetune_new_residual_net_simclr_007_80 hydra.job.name=finetune_new_residual_net_simclr_007_80  \
+onlyEval=False test_after_training=True 




CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_65_std_factor_1_Classification_Nature_60.yaml  model=my_resUnet_3D_simclr.yaml  \
+seed=100 trainer.max_epochs=100 total_folds=5  \
trainer.limit_train_batches=1.0 \
+enable_logger=False model.cfg.classify=True \
hydra.job.chdir=False  \
datamodule.cfg.batch_size=16  \
datamodule.cfg.samples_per_patient=1  \
+load_checkpoint_for_training=pretrain_simclr_t_007 \
name=finetune_new_residual_net_simclr_007_60 hydra.job.name=finetune_new_residual_net_simclr_007_60  \
+onlyEval=False test_after_training=True 



CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_65_std_factor_1_Classification_Nature_40.yaml  model=my_resUnet_3D_simclr.yaml  \
+seed=100 trainer.max_epochs=100 total_folds=5  \
trainer.limit_train_batches=1.0 \
+enable_logger=False model.cfg.classify=True \
hydra.job.chdir=False  \
datamodule.cfg.batch_size=16  \
datamodule.cfg.samples_per_patient=1  \
+load_checkpoint_for_training=pretrain_simclr_t_007 \
name=finetune_new_residual_net_simclr_007_40 hydra.job.name=finetune_new_residual_net_simclr_007_40  \
+onlyEval=False test_after_training=True 




CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_65_std_factor_1_Classification_Nature_20.yaml  model=my_resUnet_3D_simclr.yaml  \
+seed=100 trainer.max_epochs=100 total_folds=5  \
trainer.limit_train_batches=1.0 \
+enable_logger=False model.cfg.classify=True \
hydra.job.chdir=False  \
datamodule.cfg.batch_size=16  \
datamodule.cfg.samples_per_patient=1  \
+load_checkpoint_for_training=pretrain_simclr_t_007 \
name=finetune_new_residual_net_simclr_007_20 hydra.job.name=finetune_new_residual_net_simclr_007_20  \
+onlyEval=False test_after_training=True 





