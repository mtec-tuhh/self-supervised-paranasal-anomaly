




CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_65_std_factor_1_Nature_simclr_pretraining.yaml  model=my_resUnet_3D_byol.yaml  \
+seed=100 trainer.max_epochs=100 total_folds=1  \
trainer.limit_train_batches=1.0 model.cfg.classify=False   \
+enable_logger=False  \
hydra.job.chdir=False  \
datamodule.cfg.batch_size=128  \
datamodule.cfg.samples_per_patient=1  \
name=pretrain_byol hydra.job.name=pretrain_byol \
+onlyEval=False test_after_training=True 

