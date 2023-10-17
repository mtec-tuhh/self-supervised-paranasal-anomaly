

CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_55_std_factor_1_JAMA.yaml model=resnet_3D_34.yaml \
+seed=100 trainer.max_epochs=100 total_folds=5 current_fold=1 \
+enable_logger=False hydra.job.chdir=False datamodule.cfg.batch_size=1 \
datamodule.cfg.samples_per_patient=15 name=ResNet34_cc_mixed_55 hydra.job.name=ResNet34_cc_mixed_55 \
+load_checkpoint=ResNet34_cc_mixed_corrected +onlyEval=False \
test_after_training=True \
model.cfg.enable_gradcam=False


CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_55_std_factor_1_JAMA.yaml model=resnet_3D_34.yaml \
+seed=100 trainer.max_epochs=100 total_folds=5 current_fold=2 \
+enable_logger=False hydra.job.chdir=False datamodule.cfg.batch_size=1 \
datamodule.cfg.samples_per_patient=15 name=ResNet34_cc_mixed_55 hydra.job.name=ResNet34_cc_mixed_55 \
+load_checkpoint=ResNet34_cc_mixed_corrected +onlyEval=False \
test_after_training=True \
model.cfg.enable_gradcam=False



CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_55_std_factor_1_JAMA.yaml model=resnet_3D_34.yaml \
+seed=100 trainer.max_epochs=100 total_folds=5 current_fold=3 \
+enable_logger=False hydra.job.chdir=False datamodule.cfg.batch_size=1 \
datamodule.cfg.samples_per_patient=15 name=ResNet34_cc_mixed_55 hydra.job.name=ResNet34_cc_mixed_55 \
+load_checkpoint=ResNet34_cc_mixed_corrected +onlyEval=False \
test_after_training=True \
model.cfg.enable_gradcam=False




CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_55_std_factor_1_JAMA.yaml model=resnet_3D_34.yaml \
+seed=100 trainer.max_epochs=100 total_folds=5 current_fold=4 \
+enable_logger=False hydra.job.chdir=False datamodule.cfg.batch_size=1 \
datamodule.cfg.samples_per_patient=15 name=ResNet34_cc_mixed_55 hydra.job.name=ResNet34_cc_mixed_55 \
+load_checkpoint=ResNet34_cc_mixed_corrected +onlyEval=False \
test_after_training=True \
model.cfg.enable_gradcam=False




CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_55_std_factor_1_JAMA.yaml model=resnet_3D_34.yaml \
+seed=100 trainer.max_epochs=100 total_folds=5 current_fold=5 \
+enable_logger=False hydra.job.chdir=False datamodule.cfg.batch_size=1 \
datamodule.cfg.samples_per_patient=15 name=ResNet34_cc_mixed_55 hydra.job.name=ResNet34_cc_mixed_55 \
+load_checkpoint=ResNet34_cc_mixed_corrected +onlyEval=False \
test_after_training=True \
model.cfg.enable_gradcam=False
