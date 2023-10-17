


CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_35_std_factor_1.yaml model=multicontext_3D_ViT.yaml \
+seed=100 trainer.max_epochs=100 current_fold=1 \
+enable_logger=True hydra.job.chdir=False datamodule.cfg.batch_size=16 datamodule.cfg.samples_per_patient=15 \
name=multicontext_3D_ViT hydra.job.name=multicontext_3D_ViT \
+onlyEval=False test_after_training=True logger=wandb enable_convnet_model=True


