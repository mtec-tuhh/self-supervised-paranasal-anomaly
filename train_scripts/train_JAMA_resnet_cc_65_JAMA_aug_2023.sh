CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_65_std_factor_1_JAMA.yaml model=resnet_3D_50.yaml \
+seed=100 trainer.max_epochs=100 \
total_folds=5 \
+enable_logger=False \
hydra.job.chdir=False \
datamodule.cfg.batch_size=16 \
datamodule.cfg.samples_per_patient=15 \
name=ResNet50_cc_65_JAMA_aug23 \
hydra.job.name=ResNet50_cc_65_JAMA_aug23 \
+onlyEval=False test_after_training=True 
