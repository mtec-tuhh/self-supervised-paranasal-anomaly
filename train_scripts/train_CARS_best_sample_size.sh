

CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_35_std_factor_1_CARS.yaml model=resnet_3D_18.yaml \
+seed=100 trainer.max_epochs=100 \
current_fold=1 \
+enable_logger=True \
hydra.job.chdir=False \
datamodule.cfg.batch_size=15 \
datamodule.cfg.samples_per_patient=1 \
name=CARS_resnet_18_cc_35 \
hydra.job.name=CARS_resnet_18_cc_35_ss_1 \
+onlyEval=False test_after_training=True


CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_35_std_factor_1_CARS.yaml model=resnet_3D_18.yaml \
+seed=100 trainer.max_epochs=100 \
current_fold=2 \
+enable_logger=True \
hydra.job.chdir=False \
datamodule.cfg.batch_size=15 \
datamodule.cfg.samples_per_patient=1 \
name=CARS_resnet_18_cc_35 \
hydra.job.name=CARS_resnet_18_cc_35_ss_1 \
+onlyEval=False test_after_training=True



CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_35_std_factor_1_CARS.yaml model=resnet_3D_18.yaml \
+seed=100 trainer.max_epochs=100 \
current_fold=3 \
+enable_logger=True \
hydra.job.chdir=False \
datamodule.cfg.batch_size=15 \
datamodule.cfg.samples_per_patient=1 \
name=CARS_resnet_18_cc_35 \
hydra.job.name=CARS_resnet_18_cc_35_ss_1 \
+onlyEval=False test_after_training=True





CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_35_std_factor_1_CARS.yaml model=resnet_3D_18.yaml \
+seed=100 trainer.max_epochs=100 \
current_fold=1 \
+enable_logger=True \
hydra.job.chdir=False \
datamodule.cfg.batch_size=15 \
datamodule.cfg.samples_per_patient=5 \
name=CARS_resnet_18_cc_35 \
hydra.job.name=CARS_resnet_18_cc_35_ss_5 \
+onlyEval=False test_after_training=True


CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_35_std_factor_1_CARS.yaml model=resnet_3D_18.yaml \
+seed=100 trainer.max_epochs=100 \
current_fold=2 \
+enable_logger=True \
hydra.job.chdir=False \
datamodule.cfg.batch_size=15 \
datamodule.cfg.samples_per_patient=5 \
name=CARS_resnet_18_cc_35 \
hydra.job.name=CARS_resnet_18_cc_35_ss_5 \
+onlyEval=False test_after_training=True



CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_35_std_factor_1_CARS.yaml model=resnet_3D_18.yaml \
+seed=100 trainer.max_epochs=100 \
current_fold=3 \
+enable_logger=True \
hydra.job.chdir=False \
datamodule.cfg.batch_size=15 \
datamodule.cfg.samples_per_patient=5 \
name=CARS_resnet_18_cc_35 \
hydra.job.name=CARS_resnet_18_cc_35_ss_5 \
+onlyEval=False test_after_training=True






CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_35_std_factor_1_CARS.yaml model=resnet_3D_18.yaml \
+seed=100 trainer.max_epochs=100 \
current_fold=1 \
+enable_logger=True \
hydra.job.chdir=False \
datamodule.cfg.batch_size=15 \
datamodule.cfg.samples_per_patient=10 \
name=CARS_resnet_18_cc_35 \
hydra.job.name=CARS_resnet_18_cc_35_ss_10 \
+onlyEval=False test_after_training=True


CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_35_std_factor_1_CARS.yaml model=resnet_3D_18.yaml \
+seed=100 trainer.max_epochs=100 \
current_fold=2 \
+enable_logger=True \
hydra.job.chdir=False \
datamodule.cfg.batch_size=15 \
datamodule.cfg.samples_per_patient=10 \
name=CARS_resnet_18_cc_35 \
hydra.job.name=CARS_resnet_18_cc_35_ss_10 \
+onlyEval=False test_after_training=True



CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_35_std_factor_1_CARS.yaml model=resnet_3D_18.yaml \
+seed=100 trainer.max_epochs=100 \
current_fold=3 \
+enable_logger=True \
hydra.job.chdir=False \
datamodule.cfg.batch_size=15 \
datamodule.cfg.samples_per_patient=10 \
name=CARS_resnet_18_cc_35 \
hydra.job.name=CARS_resnet_18_cc_35_ss_10 \
+onlyEval=False test_after_training=True





CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_35_std_factor_1_CARS.yaml model=resnet_3D_18.yaml \
+seed=100 trainer.max_epochs=100 \
current_fold=1 \
+enable_logger=True \
hydra.job.chdir=False \
datamodule.cfg.batch_size=15 \
datamodule.cfg.samples_per_patient=15 \
name=CARS_resnet_18_cc_35 \
hydra.job.name=CARS_resnet_18_cc_35_ss_15 \
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
hydra.job.name=CARS_resnet_18_cc_35_ss_15 \
+onlyEval=False test_after_training=True



CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_35_std_factor_1_CARS.yaml model=resnet_3D_18.yaml \
+seed=100 trainer.max_epochs=100 \
current_fold=3 \
+enable_logger=True \
hydra.job.chdir=False \
datamodule.cfg.batch_size=15 \
datamodule.cfg.samples_per_patient=15 \
name=CARS_resnet_18_cc_35 \
hydra.job.name=CARS_resnet_18_cc_35_ss_15 \
+onlyEval=False test_after_training=True




CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_35_std_factor_1_CARS.yaml model=resnet_3D_18.yaml \
+seed=100 trainer.max_epochs=100 \
current_fold=1 \
+enable_logger=True \
hydra.job.chdir=False \
datamodule.cfg.batch_size=15 \
datamodule.cfg.samples_per_patient=20 \
name=CARS_resnet_18_cc_35 \
hydra.job.name=CARS_resnet_18_cc_35_ss_20 \
+onlyEval=False test_after_training=True


CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_35_std_factor_1_CARS.yaml model=resnet_3D_18.yaml \
+seed=100 trainer.max_epochs=100 \
current_fold=2 \
+enable_logger=True \
hydra.job.chdir=False \
datamodule.cfg.batch_size=15 \
datamodule.cfg.samples_per_patient=20 \
name=CARS_resnet_18_cc_35 \
hydra.job.name=CARS_resnet_18_cc_35_ss_20 \
+onlyEval=False test_after_training=True



CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_35_std_factor_1_CARS.yaml model=resnet_3D_18.yaml \
+seed=100 trainer.max_epochs=100 \
current_fold=3 \
+enable_logger=True \
hydra.job.chdir=False \
datamodule.cfg.batch_size=15 \
datamodule.cfg.samples_per_patient=20 \
name=CARS_resnet_18_cc_35 \
hydra.job.name=CARS_resnet_18_cc_35_ss_20 \
+onlyEval=False test_after_training=True