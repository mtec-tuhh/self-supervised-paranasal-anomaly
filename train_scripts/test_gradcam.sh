

#Grad-CAM enabled testing 
CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_80_std_factor_1_JAMA.yaml model=resnet_3D_34.yaml \
+seed=100 trainer.max_epochs=100 total_folds=5 current_fold=1 \
+enable_logger=False hydra.job.chdir=False datamodule.cfg.batch_size=1 \
datamodule.cfg.samples_per_patient=1 name=ResNet34_cc_80_gradcam hydra.job.name=ResNet34_cc_80_gradcam \
+load_checkpoint=ResNet34_cc_80 +onlyEval=False \
test_after_training=True \
model.cfg.enable_gradcam=True



CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_40_std_factor_1_JAMA.yaml model=resnet_3D_34.yaml \
+seed=100 trainer.max_epochs=100 total_folds=5 current_fold=1 \
+enable_logger=False hydra.job.chdir=False datamodule.cfg.batch_size=1 \
datamodule.cfg.samples_per_patient=1 name=ResNet34_cc_40_gradcam hydra.job.name=ResNet34_cc_40_gradcam \
+load_checkpoint=ResNet34_cc_40 +onlyEval=False \
test_after_training=True \
model.cfg.enable_gradcam=True


CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_45_std_factor_1_JAMA.yaml model=resnet_3D_34.yaml \
+seed=100 trainer.max_epochs=100 total_folds=5 current_fold=1 \
+enable_logger=False hydra.job.chdir=False datamodule.cfg.batch_size=1 \
datamodule.cfg.samples_per_patient=1 name=ResNet34_cc_45_gradcam hydra.job.name=ResNet34_cc_45_gradcam \
+load_checkpoint=ResNet34_cc_45 +onlyEval=False \
test_after_training=True \
model.cfg.enable_gradcam=True




CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_50_std_factor_1_JAMA.yaml model=resnet_3D_34.yaml \
+seed=100 trainer.max_epochs=100 total_folds=5 current_fold=1 \
+enable_logger=False hydra.job.chdir=False datamodule.cfg.batch_size=1 \
datamodule.cfg.samples_per_patient=1 name=ResNet34_cc_50_gradcam hydra.job.name=ResNet34_cc_50_gradcam \
+load_checkpoint=ResNet34_cc_50 +onlyEval=False \
test_after_training=True \
model.cfg.enable_gradcam=True



CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_55_std_factor_1_JAMA.yaml model=resnet_3D_34.yaml \
+seed=100 trainer.max_epochs=100 total_folds=5 current_fold=1 \
+enable_logger=False hydra.job.chdir=False datamodule.cfg.batch_size=1 \
datamodule.cfg.samples_per_patient=1 name=ResNet34_cc_55_gradcam hydra.job.name=ResNet34_cc_55_gradcam \
+load_checkpoint=ResNet34_cc_55 +onlyEval=False \
test_after_training=True \
model.cfg.enable_gradcam=True


CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_60_std_factor_1_JAMA.yaml model=resnet_3D_34.yaml \
+seed=100 trainer.max_epochs=100 total_folds=5 current_fold=1 \
+enable_logger=False hydra.job.chdir=False datamodule.cfg.batch_size=1 \
datamodule.cfg.samples_per_patient=1 name=ResNet34_cc_55_gradcam hydra.job.name=ResNet34_cc_55_gradcam \
+load_checkpoint=ResNet34_cc_55 +onlyEval=False \
test_after_training=True \
model.cfg.enable_gradcam=True



CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_65_std_factor_1_JAMA.yaml model=resnet_3D_34.yaml \
+seed=100 trainer.max_epochs=100 total_folds=5 current_fold=1 \
+enable_logger=False hydra.job.chdir=False datamodule.cfg.batch_size=1 \
datamodule.cfg.samples_per_patient=1 name=ResNet34_cc_65_gradcam hydra.job.name=ResNet34_cc_65_gradcam \
+load_checkpoint=ResNet34_cc_65 +onlyEval=False \
test_after_training=True \
model.cfg.enable_gradcam=True



CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_70_std_factor_1_JAMA.yaml model=resnet_3D_34.yaml \
+seed=100 trainer.max_epochs=100 total_folds=5 current_fold=1 \
+enable_logger=False hydra.job.chdir=False datamodule.cfg.batch_size=1 \
datamodule.cfg.samples_per_patient=1 name=ResNet34_cc_70_gradcam hydra.job.name=ResNet34_cc_70_gradcam \
+load_checkpoint=ResNet34_cc_70 +onlyEval=False \
test_after_training=True \
model.cfg.enable_gradcam=True


CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_75_std_factor_1_JAMA.yaml model=resnet_3D_34.yaml \
+seed=100 trainer.max_epochs=100 total_folds=5 current_fold=1 \
+enable_logger=False hydra.job.chdir=False datamodule.cfg.batch_size=1 \
datamodule.cfg.samples_per_patient=1 name=ResNet34_cc_75_gradcam hydra.job.name=ResNet34_cc_75_gradcam \
+load_checkpoint=ResNet34_cc_75 +onlyEval=False \
test_after_training=True \
model.cfg.enable_gradcam=True



CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_80_std_factor_1_JAMA.yaml model=resnet_3D_34.yaml \
+seed=100 trainer.max_epochs=100 total_folds=5 current_fold=1 \
+enable_logger=False hydra.job.chdir=False datamodule.cfg.batch_size=1 \
datamodule.cfg.samples_per_patient=1 name=ResNet34_cc_80_gradcam hydra.job.name=ResNet34_cc_80_gradcam \
+load_checkpoint=ResNet34_cc_80 +onlyEval=False \
test_after_training=True \
model.cfg.enable_gradcam=True