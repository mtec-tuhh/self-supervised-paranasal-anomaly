CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_15_std_factor_1_CARS.yaml model=resnet_3D_18.yaml \
+seed=100 trainer.max_epochs=100 \
current_fold=1 \
+enable_logger=False \
hydra.job.chdir=False \
datamodule.cfg.batch_size=15 \
datamodule.cfg.samples_per_patient=15 \
name=CARS_resnet_18_cc_15 \
hydra.job.name=CARS_resnet_18_cc_15 \
+onlyEval=False test_after_training=True 



CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_20_std_factor_1_CARS.yaml model=resnet_3D_18.yaml \
+seed=100 trainer.max_epochs=100 \
current_fold=1 \
+enable_logger=True \
hydra.job.chdir=False \
datamodule.cfg.batch_size=15 \
datamodule.cfg.samples_per_patient=15 \
name=CARS_resnet_18_cc_20 \
hydra.job.name=CARS_resnet_18_cc_20 \
+onlyEval=False test_after_training=True 


CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_25_std_factor_1_CARS.yaml model=resnet_3D_18.yaml \
+seed=100 trainer.max_epochs=100 \
current_fold=1 \
+enable_logger=True \
hydra.job.chdir=False \
datamodule.cfg.batch_size=15 \
datamodule.cfg.samples_per_patient=15 \
name=CARS_resnet_18_cc_25 \
hydra.job.name=CARS_resnet_18_cc_25 \
+onlyEval=False test_after_training=True 


CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_30_std_factor_1_CARS.yaml model=resnet_3D_18.yaml \
+seed=100 trainer.max_epochs=100 \
current_fold=1 \
+enable_logger=True \
hydra.job.chdir=False \
datamodule.cfg.batch_size=15 \
datamodule.cfg.samples_per_patient=15 \
name=CARS_resnet_18_cc_30 \
hydra.job.name=CARS_resnet_18_cc_30 \
+onlyEval=False test_after_training=True


CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_35_std_factor_1_CARS.yaml model=resnet_3D_18.yaml \
+seed=100 trainer.max_epochs=100 \
current_fold=2 \
+enable_logger=True \
hydra.job.chdir=False \
datamodule.cfg.batch_size=15 \
datamodule.cfg.samples_per_patient=15 \
name=CARS_resnet_18_cc_35 \
hydra.job.name=CARS_resnet_18_cc_35 \
+onlyEval=False test_after_training=True




CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_40_std_factor_1_CARS.yaml model=resnet_3D_18.yaml \
+seed=100 trainer.max_epochs=100 \
current_fold=1 \
+enable_logger=True \
hydra.job.chdir=False \
datamodule.cfg.batch_size=15 \
datamodule.cfg.samples_per_patient=15 \
name=CARS_resnet_18_cc_40 \
hydra.job.name=CARS_resnet_18_cc_40 \
+onlyEval=False test_after_training=True



CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_45_std_factor_1_CARS.yaml model=resnet_3D_18.yaml \
+seed=100 trainer.max_epochs=100 \
current_fold=1 \
+enable_logger=True \
hydra.job.chdir=False \
datamodule.cfg.batch_size=15 \
datamodule.cfg.samples_per_patient=15 \
name=CARS_resnet_18_cc_45 \
hydra.job.name=CARS_resnet_18_cc_45 \
+onlyEval=False test_after_training=True




CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_50_std_factor_1_CARS.yaml model=resnet_3D_18.yaml \
+seed=100 trainer.max_epochs=100 \
current_fold=1 \
+enable_logger=True \
hydra.job.chdir=False \
datamodule.cfg.batch_size=15 \
datamodule.cfg.samples_per_patient=15 \
name=CARS_resnet_18_cc_50 \
hydra.job.name=CARS_resnet_18_cc_50 \
+onlyEval=False test_after_training=True



CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_55_std_factor_1_CARS.yaml model=resnet_3D_18.yaml \
+seed=100 trainer.max_epochs=100 \
current_fold=1 \
+enable_logger=True \
hydra.job.chdir=False \
datamodule.cfg.batch_size=15 \
datamodule.cfg.samples_per_patient=15 \
name=CARS_resnet_18_cc_55 \
hydra.job.name=CARS_resnet_18_cc_55 \
+onlyEval=False test_after_training=True








