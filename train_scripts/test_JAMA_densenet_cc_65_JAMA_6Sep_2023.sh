


CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_65_std_factor_1_JAMA.yaml model=densenet_3D_264.yaml \
+seed=100 trainer.max_epochs=100 total_folds=5 current_fold=3 \
+enable_logger=False hydra.job.chdir=False datamodule.cfg.batch_size=1 \
datamodule.cfg.samples_per_patient=1 name=DenseNet264_cc_65_JAMA_aug23 hydra.job.name=DenseNet264_cc_65_JAMA_aug23 \
+load_checkpoint=DenseNet264_cc_65_JAMA_aug23 +onlyEval=False \
test_after_training=True \
model.cfg.enable_gradcam=True


