
CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_35_std_factor_1_CARS_yz.yaml model=resnet_3D_18.yaml \
+seed=100 trainer.max_epochs=100 \
total_folds=3 \
+enable_logger=False \
hydra.job.chdir=False \
datamodule.cfg.batch_size=16 \
datamodule.cfg.samples_per_patient=15 \
name=ResNet18_cc_60_CARS_yz_15 \
hydra.job.name=ResNet18_cc_60_CARS_yz_15 \
+onlyEval=False test_after_training=True 





CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_35_std_factor_1_CARS_yz.yaml model=resnet_3D_18.yaml \
+seed=100 trainer.max_epochs=100 \
total_folds=3 \
+enable_logger=False \
hydra.job.chdir=False \
datamodule.cfg.batch_size=16 \
datamodule.cfg.samples_per_patient=1 \
name=ResNet18_cc_60_CARS_yz_1 \
hydra.job.name=ResNet18_cc_60_CARS_yz_1 \
+onlyEval=False test_after_training=True 


CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_35_std_factor_1_CARS_yz.yaml model=resnet_3D_18.yaml \
+seed=100 trainer.max_epochs=100 \
total_folds=3 \
+enable_logger=False \
hydra.job.chdir=False \
datamodule.cfg.batch_size=16 \
datamodule.cfg.samples_per_patient=5 \
name=ResNet18_cc_60_CARS_yz_5 \
hydra.job.name=ResNet18_cc_60_CARS_yz_5 \
+onlyEval=False test_after_training=True 



CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_35_std_factor_1_CARS_yz.yaml model=resnet_3D_18.yaml \
+seed=100 trainer.max_epochs=100 \
total_folds=3 \
+enable_logger=False \
hydra.job.chdir=False \
datamodule.cfg.batch_size=16 \
datamodule.cfg.samples_per_patient=10 \
name=ResNet18_cc_60_CARS_yz_10 \
hydra.job.name=ResNet18_cc_60_CARS_yz_10 \
+onlyEval=False test_after_training=True 

