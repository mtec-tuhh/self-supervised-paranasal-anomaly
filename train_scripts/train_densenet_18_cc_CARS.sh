

CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_35_std_factor_1_CARS.yaml model=densenet_3D_121.yaml \
+seed=100 trainer.max_epochs=100 \
total_folds=3 \
+enable_logger=False \
hydra.job.chdir=False \
datamodule.cfg.batch_size=16 \
datamodule.cfg.samples_per_patient=15 \
name=DenseNet121_cc_CARS_x_15_ss_15 \
hydra.job.name=DenseNet121_cc_CARS_x_15_ss_15 \
+onlyEval=False test_after_training=True 


CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_35_std_factor_1_CARS.yaml model=densenet_3D_169.yaml \
+seed=100 trainer.max_epochs=100 \
total_folds=3 \
+enable_logger=False \
hydra.job.chdir=False \
datamodule.cfg.batch_size=16 \
datamodule.cfg.samples_per_patient=15 \
name=DenseNet169_cc_CARS_x_15_ss_15 \
hydra.job.name=DenseNet169_cc_CARS_x_15_ss_15 \
+onlyEval=False test_after_training=True 



CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_35_std_factor_1_CARS.yaml model=densenet_3D_201.yaml \
+seed=100 trainer.max_epochs=100 \
total_folds=3 \
+enable_logger=False \
hydra.job.chdir=False \
datamodule.cfg.batch_size=16 \
datamodule.cfg.samples_per_patient=15 \
name=DenseNet201_cc_CARS_x_15_ss_15 \
hydra.job.name=DenseNet201_cc_CARS_x_15_ss_15 \
+onlyEval=False test_after_training=True 


CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_35_std_factor_1_CARS.yaml model=densenet_3D_264.yaml \
+seed=100 trainer.max_epochs=100 \
total_folds=3 \
+enable_logger=False \
hydra.job.chdir=False \
datamodule.cfg.batch_size=16 \
datamodule.cfg.samples_per_patient=15 \
name=DenseNet264_cc_CARS_x_15_ss_15 \
hydra.job.name=DenseNet264_cc_CARS_x_15_ss_15 \
+onlyEval=False test_after_training=True 

