CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run.py \
datamodule=crop_size_65_std_factor_1_JAMA.yaml model=densenet_3D_264.yaml \
+seed=100 trainer.max_epochs=100 trainer.limit_train_batches=1.0 trainer.limit_val_batches=1.0 trainer.limit_test_batches=1.0 \
total_folds=5 \
+enable_logger=False \
hydra.job.chdir=False \
datamodule.cfg.batch_size=16 \
datamodule.cfg.samples_per_patient=15 \
name=DenseNet264_cc_65_JAMA_6Sep23 \
hydra.job.name=DenseNet264_cc_65_JAMA_6Sep23 \
+onlyEval=False test_after_training=True 