CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_35_std_factor_1.yaml model=trans_resnet_3D_18.yaml \
+seed=100 trainer.max_epochs=100 \
current_fold=3 \
+enable_logger=True trainer.limit_train_batches=1.0  \
hydra.job.chdir=False \
datamodule.cfg.batch_size=16 \
datamodule.cfg.samples_per_patient=1 \
name=resnet_18_samplesize_1 \
model.cfg.enable_transformer=False \
hydra.job.name=resnet_18_samplesize_1 \
+onlyEval=False test_after_training=True 

CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_35_std_factor_1.yaml model=trans_resnet_3D_18.yaml \
+seed=100 trainer.max_epochs=100 \
current_fold=3 \
+enable_logger=True trainer.limit_train_batches=1.0  \
hydra.job.chdir=False \
datamodule.cfg.batch_size=16 \
datamodule.cfg.samples_per_patient=5 \
name=resnet_18_samplesize_5 \
model.cfg.enable_transformer=False \
hydra.job.name=resnet_18_samplesize_5 \
+onlyEval=False test_after_training=True 

CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_35_std_factor_1.yaml model=trans_resnet_3D_18.yaml \
+seed=100 trainer.max_epochs=100 \
current_fold=3 \
+enable_logger=True trainer.limit_train_batches=1.0  \
hydra.job.chdir=False \
datamodule.cfg.batch_size=16 \
datamodule.cfg.samples_per_patient=10 \
name=resnet_18_samplesize_10 \
model.cfg.enable_transformer=False \
hydra.job.name=resnet_18_samplesize_10 \
+onlyEval=False test_after_training=True 







CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_35_std_factor_1.yaml model=trans_resnet_3D_34.yaml \
+seed=100 trainer.max_epochs=100 \
current_fold=3 \
+enable_logger=True trainer.limit_train_batches=1.0  \
hydra.job.chdir=False \
datamodule.cfg.batch_size=16 \
datamodule.cfg.samples_per_patient=1 \
name=resnet_34_samplesize_1 \
model.cfg.enable_transformer=False \
hydra.job.name=resnet_34_samplesize_1 \
+onlyEval=False test_after_training=True 

CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_35_std_factor_1.yaml model=trans_resnet_3D_34.yaml \
+seed=100 trainer.max_epochs=100 \
current_fold=3 \
+enable_logger=True trainer.limit_train_batches=1.0  \
hydra.job.chdir=False \
datamodule.cfg.batch_size=16 \
datamodule.cfg.samples_per_patient=5 \
name=resnet_34_samplesize_5 \
model.cfg.enable_transformer=False \
hydra.job.name=resnet_34_samplesize_5 \
+onlyEval=False test_after_training=True 

CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_35_std_factor_1.yaml model=trans_resnet_3D_34.yaml \
+seed=100 trainer.max_epochs=100 \
current_fold=3 \
+enable_logger=True trainer.limit_train_batches=1.0  \
hydra.job.chdir=False \
datamodule.cfg.batch_size=16 \
datamodule.cfg.samples_per_patient=10 \
name=resnet_34_samplesize_10 \
model.cfg.enable_transformer=False \
hydra.job.name=resnet_34_samplesize_10 \
+onlyEval=False test_after_training=True 


CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_35_std_factor_1.yaml model=trans_resnet_3D_50.yaml \
+seed=100 trainer.max_epochs=100 \
current_fold=3 \
+enable_logger=True trainer.limit_train_batches=1.0  \
hydra.job.chdir=False \
datamodule.cfg.batch_size=16 \
datamodule.cfg.samples_per_patient=1 \
name=resnet_50_samplesize_1 \
model.cfg.enable_transformer=False \
hydra.job.name=resnet_50_samplesize_1 \
+onlyEval=False test_after_training=True 

CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_35_std_factor_1.yaml model=trans_resnet_3D_50.yaml \
+seed=100 trainer.max_epochs=100 \
current_fold=3 \
+enable_logger=True trainer.limit_train_batches=1.0  \
hydra.job.chdir=False \
datamodule.cfg.batch_size=16 \
datamodule.cfg.samples_per_patient=5 \
name=resnet_50_samplesize_5 \
model.cfg.enable_transformer=False \
hydra.job.name=resnet_50_samplesize_5 \
+onlyEval=False test_after_training=True 

CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_35_std_factor_1.yaml model=trans_resnet_3D_50.yaml \
+seed=100 trainer.max_epochs=100 \
current_fold=3 \
+enable_logger=True trainer.limit_train_batches=1.0  \
hydra.job.chdir=False \
datamodule.cfg.batch_size=16 \
datamodule.cfg.samples_per_patient=10 \
name=resnet_50_samplesize_10 \
model.cfg.enable_transformer=False \
hydra.job.name=resnet_50_samplesize_10 \
+onlyEval=False test_after_training=True 