

#Grad-CAM enabled testing 
CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python run.py  \
datamodule=crop_size_65_std_factor_1_Nature_DAE.yaml model=ae_3D.yaml \
+seed=100 trainer.max_epochs=100 total_folds=5 current_fold=1 model.cfg.save_folder=AE_3D_cc_65_ls_512_20 \
+enable_logger=False hydra.job.chdir=False datamodule.cfg.batch_size=1 \
datamodule.cfg.samples_per_patient=1 name=AE_3D_cc_65_ls_512_20 hydra.job.name=AE_3D_cc_65_ls_512_20 \
+load_checkpoint=AE_3D_cc_65_ls_512_20 +onlyEval=False \
test_after_training=True \
trainer.max_epochs=100


CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_65_std_factor_1_Nature_DAE.yaml model=ae_3D.yaml \
+seed=100 trainer.max_epochs=100 total_folds=5 current_fold=1 \
+enable_logger=False hydra.job.chdir=False datamodule.cfg.batch_size=1 model.cfg.save_folder=AE_3D_cc_65_ls_512_40 \
datamodule.cfg.samples_per_patient=1 name=AE_3D_cc_65_ls_512_40 hydra.job.name=AE_3D_cc_65_ls_512_40 \
+load_checkpoint=AE_3D_cc_65_ls_512_40 +onlyEval=False \
test_after_training=True \
trainer.max_epochs=100


CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_65_std_factor_1_Nature_DAE.yaml model=ae_3D.yaml \
+seed=100 trainer.max_epochs=100 total_folds=5 current_fold=1 \
+enable_logger=False hydra.job.chdir=False datamodule.cfg.batch_size=1 model.cfg.save_folder=AE_3D_cc_65_ls_512_60 \
datamodule.cfg.samples_per_patient=1 name=AE_3D_cc_65_ls_512_60 hydra.job.name=AE_3D_cc_65_ls_512_60 \
+load_checkpoint=AE_3D_cc_65_ls_512_60 +onlyEval=False \
test_after_training=True \
trainer.max_epochs=100



CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_65_std_factor_1_Nature_DAE.yaml model=ae_3D.yaml \
+seed=100 trainer.max_epochs=100 total_folds=5 current_fold=1 \
+enable_logger=False hydra.job.chdir=False datamodule.cfg.batch_size=1 model.cfg.save_folder=AE_3D_cc_65_ls_512_80 \
datamodule.cfg.samples_per_patient=1 name=AE_3D_cc_65_ls_512_80 hydra.job.name=AE_3D_cc_65_ls_512_80 \
+load_checkpoint=AE_3D_cc_65_ls_512_80 +onlyEval=False \
test_after_training=True \
trainer.max_epochs=100

