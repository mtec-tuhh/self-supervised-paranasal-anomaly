CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_35_std_factor_1.yaml model=trans_resnet_3D_18.yaml \
+seed=100 trainer.max_epochs=100 \
current_fold=1 \
+enable_logger=False \
hydra.job.chdir=False \
datamodule.cfg.batch_size=16 \
datamodule.cfg.samples_per_patient=1 \
name=contra-net-s-ss-1-nocls \
hydra.job.name=contra-net-s-ss-1-nocls \
model.cfg.enable_transformer=True \
+onlyEval=False test_after_training=True 

CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_35_std_factor_1.yaml model=trans_resnet_3D_18.yaml \
+seed=100 trainer.max_epochs=100 \
current_fold=2 \
+enable_logger=False \
hydra.job.chdir=False \
datamodule.cfg.batch_size=16 \
datamodule.cfg.samples_per_patient=1 \
name=contra-net-s-ss-1-nocls \
hydra.job.name=contra-net-s-ss-1-nocls \
model.cfg.enable_transformer=True \
+onlyEval=False test_after_training=True 

CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_35_std_factor_1.yaml model=trans_resnet_3D_18.yaml \
+seed=100 trainer.max_epochs=100 \
current_fold=3 \
+enable_logger=False \
hydra.job.chdir=False \
datamodule.cfg.batch_size=16 \
datamodule.cfg.samples_per_patient=1 \
name=contra-net-s-ss-1-nocls \
model.cfg.enable_transformer=True \
hydra.job.name=contra-net-s-ss-1-nocls \
+onlyEval=False test_after_training=True 


