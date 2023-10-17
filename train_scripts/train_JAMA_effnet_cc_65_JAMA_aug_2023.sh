CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_65_std_factor_1_JAMA.yaml model=efficientnet_3D.yaml \
+seed=100 trainer.max_epochs=100 model.cfg.model_name=efficientnet-b6 \
total_folds=5 \
+enable_logger=False \
hydra.job.chdir=False \
datamodule.cfg.batch_size=16 \
datamodule.cfg.samples_per_patient=15 \
name=EffNetB5_cc_65_JAMA_aug23 \
hydra.job.name=EffNetB5_cc_65_JAMA_aug23 \
+onlyEval=False test_after_training=True 
