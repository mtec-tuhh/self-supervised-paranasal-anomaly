

CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_60_std_factor_1_JAMA.yaml model=resnet_3D_18.yaml \
+seed=100 trainer.max_epochs=100 \
total_folds=5 \
+enable_logger=False \
hydra.job.chdir=False \
datamodule.cfg.batch_size=16 \
datamodule.cfg.samples_per_patient=15 \
name=ResNet18_cc_60 \
hydra.job.name=ResNet18_cc_60 \
+onlyEval=False test_after_training=True 



CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_65_std_factor_1_JAMA.yaml model=resnet_3D_18.yaml \
+seed=100 trainer.max_epochs=100 \
total_folds=5 \
+enable_logger=False \
hydra.job.chdir=False \
datamodule.cfg.batch_size=16 \
datamodule.cfg.samples_per_patient=15 \
name=ResNet18_cc_65 \
hydra.job.name=ResNet18_cc_65 \
+onlyEval=False test_after_training=True 


CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_70_std_factor_1_JAMA.yaml model=resnet_3D_18.yaml \
+seed=100 trainer.max_epochs=100 \
total_folds=5 \
+enable_logger=False \
hydra.job.chdir=False \
datamodule.cfg.batch_size=16 \
datamodule.cfg.samples_per_patient=15 \
name=ResNet18_cc_70 \
hydra.job.name=ResNet18_cc_70 \
+onlyEval=False test_after_training=True 



CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_75_std_factor_1_JAMA.yaml model=resnet_3D_18.yaml \
+seed=100 trainer.max_epochs=100 \
total_folds=5 \
+enable_logger=False \
hydra.job.chdir=False \
datamodule.cfg.batch_size=16 \
datamodule.cfg.samples_per_patient=15 \
name=ResNet18_cc_75 \
hydra.job.name=ResNet18_cc_75 \
+onlyEval=False test_after_training=True 


CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_80_std_factor_1_JAMA.yaml model=resnet_3D_18.yaml \
+seed=100 trainer.max_epochs=100 \
total_folds=5 \
+enable_logger=False \
hydra.job.chdir=False \
datamodule.cfg.batch_size=16 \
datamodule.cfg.samples_per_patient=15 \
name=ResNet18_cc_80 \
hydra.job.name=ResNet18_cc_80 \
+onlyEval=False test_after_training=True 

