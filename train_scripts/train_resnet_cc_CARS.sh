

CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_35_std_factor_1_CARS.yaml model=resnet_3D_18.yaml model.cfg.model_depth=50 \
+seed=100 trainer.max_epochs=100 \
total_folds=3 \
+enable_logger=False \
hydra.job.chdir=False \
datamodule.cfg.batch_size=16 \
datamodule.cfg.samples_per_patient=1 \
name=ResNet50_cc_CARS_x_15 \
hydra.job.name=ResNet50_cc_CARS_x \
+onlyEval=False test_after_training=True 


CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_35_std_factor_1_CARS.yaml model=resnet_3D_18.yaml model.cfg.model_depth=50 \
+seed=100 trainer.max_epochs=100 \
total_folds=3 \
+enable_logger=False \
hydra.job.chdir=False \
datamodule.cfg.batch_size=16 \
datamodule.cfg.samples_per_patient=15 \
name=ResNet50_cc_CARS_x_15_ss_15 \
hydra.job.name=ResNet50_cc_CARS_x_15_ss_15 \
+onlyEval=False test_after_training=True 


CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_35_std_factor_1_CARS.yaml model=resnet_3D_18.yaml model.cfg.model_depth=101 \
+seed=100 trainer.max_epochs=100 \
total_folds=3 \
+enable_logger=False \
hydra.job.chdir=False \
datamodule.cfg.batch_size=16 \
datamodule.cfg.samples_per_patient=1 \
name=ResNet101_ccc_CARS_x_15 \
hydra.job.name=ResNet101_ccc_CARS_x_15 \
+onlyEval=False test_after_training=True 

CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_35_std_factor_1_CARS.yaml model=resnet_3D_18.yaml model.cfg.model_depth=101 \
+seed=100 trainer.max_epochs=100 \
total_folds=3 \
+enable_logger=False \
hydra.job.chdir=False \
datamodule.cfg.batch_size=16 \
datamodule.cfg.samples_per_patient=15 \
name=ResNet101_ccc_CARS_x_15_ss_15 \
hydra.job.name=ResNet101_ccc_CARS_x_15_ss_15 \
+onlyEval=False test_after_training=True 

CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_35_std_factor_1_CARS.yaml model=resnet_3D_18.yaml model.cfg.model_depth=152 \
+seed=100 trainer.max_epochs=100 \
total_folds=3 \
+enable_logger=False \
hydra.job.chdir=False \
datamodule.cfg.batch_size=16 \
datamodule.cfg.samples_per_patient=1 \
name=ResNet152_cc_CARS_x_15 \
hydra.job.name=ResNet152_cc_CARS_x_15 \
+onlyEval=False test_after_training=True 


CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_35_std_factor_1_CARS.yaml model=resnet_3D_18.yaml model.cfg.model_depth=152 \
+seed=100 trainer.max_epochs=100 \
total_folds=3 \
+enable_logger=False \
hydra.job.chdir=False \
datamodule.cfg.batch_size=16 \
datamodule.cfg.samples_per_patient=15 \
name=ResNet152_cc_CARS_x_15_ss_15 \
hydra.job.name=ResNet152_cc_CARS_x_15_ss_15 \
+onlyEval=False test_after_training=True 


CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_35_std_factor_1_CARS.yaml model=resnet_3D_18.yaml model.cfg.model_depth=200 \
+seed=100 trainer.max_epochs=100 \
total_folds=3 \
+enable_logger=False \
hydra.job.chdir=False \
datamodule.cfg.batch_size=16 \
datamodule.cfg.samples_per_patient=1 \
name=ResNet200_cc_CARS_x_15 \
hydra.job.name=ResNet200_cc_CARS_x_15 \
+onlyEval=False test_after_training=True 


CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_35_std_factor_1_CARS.yaml model=resnet_3D_18.yaml model.cfg.model_depth=200 \
+seed=100 trainer.max_epochs=100 \
total_folds=3 \
+enable_logger=False \
hydra.job.chdir=False \
datamodule.cfg.batch_size=16 \
datamodule.cfg.samples_per_patient=15 \
name=ResNet200_cc_CARS_x_15_ss_15 \
hydra.job.name=ResNet200_cc_CARS_x_15_ss_15 \
+onlyEval=False test_after_training=True 

