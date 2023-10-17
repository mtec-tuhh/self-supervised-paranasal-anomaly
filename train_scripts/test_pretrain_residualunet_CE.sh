

#Grad-CAM enabled testing 
CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python run.py  \
datamodule=crop_size_65_std_factor_1_Nature_Residual_pretraining.yaml model=my_resUnet_3D.yaml model.cfg.loss=ce \
+seed=100 trainer.max_epochs=100 total_folds=5 current_fold=1 model.cfg.save_folder=test_l1 \
+enable_logger=False hydra.job.chdir=False datamodule.cfg.batch_size=1 \
datamodule.cfg.samples_per_patient=1 name=test_l1 hydra.job.name=test_l1 \
+load_checkpoint=pretrain_LARS_residual_net_bs_128_l1_MF_5 +onlyEval=False \
datamodule.cfg.median_filter=MF_5  \
test_after_training=True \
trainer.max_epochs=100
