
CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_35_std_factor_1_CARS.yaml model=efficientnet_3D.yaml \
+seed=100 trainer.max_epochs=100 model.cfg.model_name=efficientnet-b4 \
total_folds=3 \
+enable_logger=False \
hydra.job.chdir=False \
datamodule.cfg.batch_size=16 \
datamodule.cfg.samples_per_patient=1 \
name=EffNetB4_cc_CARS_x_15 \
hydra.job.name=EffNetB4_cc_CARS_x_15 \
+onlyEval=False test_after_training=True 


CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_35_std_factor_1_CARS.yaml model=efficientnet_3D.yaml \
+seed=100 trainer.max_epochs=100 model.cfg.model_name=efficientnet-b5 \
total_folds=3 \
+enable_logger=False \
hydra.job.chdir=False \
datamodule.cfg.batch_size=16 \
datamodule.cfg.samples_per_patient=15 \
name=EffNetB5_cc_CARS_x_15_ss_15 \
hydra.job.name=EffNetB5_cc_CARS_x_15_ss_15 \
+onlyEval=False test_after_training=True 
