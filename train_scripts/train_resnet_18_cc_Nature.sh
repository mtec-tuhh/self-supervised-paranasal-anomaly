

CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run.py \
datamodule=crop_size_65_std_factor_1_Classification_Nature.yaml  \
model=my_resUnet_3D.yaml  +seed=100 trainer.max_epochs=100 \
total_folds=5  trainer.limit_train_batches=1.0 \
model.cfg.classify=True  +enable_logger=False  \
hydra.job.chdir=False  datamodule.cfg.batch_size=16  \
datamodule.cfg.samples_per_patient=1 \
name=no_pretrain_residual_net_20 \
trainer.limit_train_batches=0.2 \
hydra.job.name=no_pretrain_residual_net_20  \
+onlyEval=False test_after_training=True 


CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run.py \
datamodule=crop_size_65_std_factor_1_Classification_Nature.yaml  \
model=my_resUnet_3D.yaml  +seed=100 trainer.max_epochs=100 \
total_folds=5  trainer.limit_train_batches=1.0 \
model.cfg.classify=True  +enable_logger=False  \
hydra.job.chdir=False  datamodule.cfg.batch_size=16  \
datamodule.cfg.samples_per_patient=1 \
name=no_pretrain_residual_net_40 \
trainer.limit_train_batches=0.4 \
hydra.job.name=no_pretrain_residual_net_40  \
+onlyEval=False test_after_training=True 


CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run.py \
datamodule=crop_size_65_std_factor_1_Classification_Nature.yaml  \
model=my_resUnet_3D.yaml  +seed=100 trainer.max_epochs=100 \
total_folds=5  trainer.limit_train_batches=1.0 \
model.cfg.classify=True  +enable_logger=False  \
hydra.job.chdir=False  datamodule.cfg.batch_size=16  \
datamodule.cfg.samples_per_patient=1 \
name=no_pretrain_residual_net_60 \
trainer.limit_train_batches=0.6 \
hydra.job.name=no_pretrain_residual_net_60  \
+onlyEval=False test_after_training=True 




CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run.py \
datamodule=crop_size_65_std_factor_1_Classification_Nature.yaml  \
model=my_resUnet_3D.yaml  +seed=100 trainer.max_epochs=100 \
total_folds=5  trainer.limit_train_batches=1.0 \
model.cfg.classify=True  +enable_logger=False  \
hydra.job.chdir=False  datamodule.cfg.batch_size=16  \
datamodule.cfg.samples_per_patient=1 \
name=no_pretrain_residual_net_80 \
trainer.limit_train_batches=0.8 \
hydra.job.name=no_pretrain_residual_net_80  \
+onlyEval=False test_after_training=True 