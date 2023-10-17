CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python run.py  \
datamodule=crop_size_35_std_factor_1.yaml model=trans_resnet_3D_18.yaml \
+seed=100 trainer.max_epochs=100 \
current_fold=1 \
+enable_logger=True \
hydra.job.chdir=False \
datamodule.cfg.batch_size=16 \
datamodule.cfg.samples_per_patient=15 \
name=resnet_18 \
model.cfg.enable_transformer=False \
hydra.job.name=resnet_18 \
+onlyEval=False test_after_training=True 

CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python run.py  \
datamodule=crop_size_35_std_factor_1.yaml model=trans_resnet_3D_18.yaml \
+seed=100 trainer.max_epochs=100 \
current_fold=2 \
+enable_logger=True \
hydra.job.chdir=False \
datamodule.cfg.batch_size=16 \
datamodule.cfg.samples_per_patient=15 \
name=resnet_18 \
model.cfg.enable_transformer=False \
hydra.job.name=resnet_18 \
+onlyEval=False test_after_training=True 

CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python run.py  \
datamodule=crop_size_35_std_factor_1.yaml model=trans_resnet_3D_18.yaml \
+seed=100 trainer.max_epochs=100 \
current_fold=3 \
+enable_logger=True \
hydra.job.chdir=False \
datamodule.cfg.batch_size=16 \
datamodule.cfg.samples_per_patient=15 \
name=resnet_18 \
model.cfg.enable_transformer=False \
hydra.job.name=resnet_18 \
+onlyEval=False test_after_training=True 







CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python run.py  \
datamodule=crop_size_35_std_factor_1.yaml model=trans_resnet_3D_34.yaml \
+seed=100 trainer.max_epochs=100 \
current_fold=1 \
+enable_logger=True \
hydra.job.chdir=False \
datamodule.cfg.batch_size=16 \
datamodule.cfg.samples_per_patient=15 \
name=resnet_34 \
model.cfg.enable_transformer=False \
hydra.job.name=resnet_34 \
+onlyEval=False test_after_training=True 

CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python run.py  \
datamodule=crop_size_35_std_factor_1.yaml model=trans_resnet_3D_34.yaml \
+seed=100 trainer.max_epochs=100 \
current_fold=2 \
+enable_logger=True \
hydra.job.chdir=False \
datamodule.cfg.batch_size=16 \
datamodule.cfg.samples_per_patient=15 \
name=resnet_34 \
model.cfg.enable_transformer=False \
hydra.job.name=resnet_34 \
+onlyEval=False test_after_training=True 

CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python run.py  \
datamodule=crop_size_35_std_factor_1.yaml model=trans_resnet_3D_34.yaml \
+seed=100 trainer.max_epochs=100 \
current_fold=3 \
+enable_logger=True \
hydra.job.chdir=False \
datamodule.cfg.batch_size=16 \
datamodule.cfg.samples_per_patient=15 \
name=resnet_34 \
model.cfg.enable_transformer=False \
hydra.job.name=resnet_34 \
+onlyEval=False test_after_training=True 









CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python run.py  \
datamodule=crop_size_35_std_factor_1.yaml model=trans_resnet_3D_50.yaml \
+seed=100 trainer.max_epochs=100 \
current_fold=1 \
+enable_logger=True \
hydra.job.chdir=False \
datamodule.cfg.batch_size=16 \
datamodule.cfg.samples_per_patient=15 \
name=resnet_50 \
model.cfg.enable_transformer=False \
hydra.job.name=resnet_50 \
+onlyEval=False test_after_training=True 

CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python run.py  \
datamodule=crop_size_35_std_factor_1.yaml model=trans_resnet_3D_50.yaml \
+seed=100 trainer.max_epochs=100 \
current_fold=2 \
+enable_logger=True \
hydra.job.chdir=False \
datamodule.cfg.batch_size=16 \
datamodule.cfg.samples_per_patient=15 \
name=resnet_50 \
model.cfg.enable_transformer=False \
hydra.job.name=resnet_50 \
+onlyEval=False test_after_training=True 

CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python run.py  \
datamodule=crop_size_35_std_factor_1.yaml model=trans_resnet_3D_50.yaml \
+seed=100 trainer.max_epochs=100 \
current_fold=3 \
+enable_logger=True \
hydra.job.chdir=False \
datamodule.cfg.batch_size=16 \
datamodule.cfg.samples_per_patient=15 \
name=resnet_50 \
model.cfg.enable_transformer=False \
hydra.job.name=resnet_50 \
+onlyEval=False test_after_training=True 