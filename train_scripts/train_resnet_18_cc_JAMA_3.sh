

CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_mixed_std_factor_1_JAMA.yaml model=resnet_3D_18.yaml \
+seed=100 trainer.max_epochs=100 \
total_folds=5 \
+enable_logger=False \
hydra.job.chdir=False \
datamodule.cfg.batch_size=16 \
datamodule.cfg.samples_per_patient=15 \
name=ResNet18_cc_mixed \
hydra.job.name=ResNet18_cc_mixed \
+onlyEval=False test_after_training=True 


