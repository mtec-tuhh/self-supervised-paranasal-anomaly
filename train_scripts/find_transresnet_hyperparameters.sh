CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python run.py  \
datamodule=crop_size_35_std_factor_1.yaml model=trans_resnet_3D_18.yaml \
+seed=100 trainer.max_epochs=100 \
current_fold=1 \
+enable_logger=True \
hydra.job.chdir=False \
datamodule.cfg.batch_size=16 \
datamodule.cfg.samples_per_patient=15 \
model.cfg.t_depth=2 \
model.cfg.t_heads=8 \
name=trans_resnet_18_d_2_h_8 \
hydra.job.name=trans_resnet_d_2_h_8 \
+onlyEval=False test_after_training=True 





CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python run.py  \
datamodule=crop_size_35_std_factor_1.yaml model=trans_resnet_3D_18.yaml \
+seed=100 trainer.max_epochs=100 \
current_fold=1 \
+enable_logger=True \
hydra.job.chdir=False \
datamodule.cfg.batch_size=16 \
datamodule.cfg.samples_per_patient=15 \
model.cfg.t_depth=4 \
model.cfg.t_heads=4 \
name=trans_resnet_18_d_4_h_4 \
hydra.job.name=trans_resnet_18_d_4_4 \
+onlyEval=False test_after_training=True 



CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python run.py  \
datamodule=crop_size_35_std_factor_1.yaml model=trans_resnet_3D_18.yaml \
+seed=100 trainer.max_epochs=100 \
current_fold=1 \
+enable_logger=True \
hydra.job.chdir=False \
datamodule.cfg.batch_size=16 \
datamodule.cfg.samples_per_patient=15 \
model.cfg.t_depth=6 \
model.cfg.t_heads=6 \
name=trans_resnet_18_d_6_h_6 \
hydra.job.name=trans_resnet_18_d_6_h_6 \
+onlyEval=False test_after_training=True 


CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python run.py  \
datamodule=crop_size_35_std_factor_1.yaml model=trans_resnet_3D_18.yaml \
+seed=100 trainer.max_epochs=100 \
current_fold=1 \
+enable_logger=True \
hydra.job.chdir=False \
datamodule.cfg.batch_size=16 \
datamodule.cfg.samples_per_patient=15 \
model.cfg.t_depth=8 \
model.cfg.t_heads=8 \
name=trans_resnet_18_d_8 \
hydra.job.name=trans_resnet_18_d_8_h_8 \
+onlyEval=False test_after_training=True 