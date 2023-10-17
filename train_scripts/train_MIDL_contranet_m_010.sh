

#ContraNet 
CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_35_std_factor_1.yaml model=trans_resnet_3D_34.yaml \
+seed=100 trainer.max_epochs=100 \
current_fold=1 \
+enable_logger=False \
hydra.job.chdir=False \
datamodule.cfg.batch_size=16 \
datamodule.cfg.samples_per_patient=15 \
model.cfg.features_to_include=[0,1,0] \
name=midlrebuttal_contranet-m-010\
model.cfg.enable_transformer=True \
hydra.job.name=midlrebuttal_contranet-m-010 \
+onlyEval=False test_after_training=True 


CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_35_std_factor_1.yaml model=trans_resnet_3D_34.yaml \
+seed=100 trainer.max_epochs=100 \
current_fold=2 \
+enable_logger=False \
hydra.job.chdir=False \
datamodule.cfg.batch_size=16 \
datamodule.cfg.samples_per_patient=15 \
model.cfg.features_to_include=[0,1,0] \
name=midlrebuttal_contranet-m-010 \
model.cfg.enable_transformer=True \
hydra.job.name=midlrebuttal_contranet-m-010 \
+onlyEval=False test_after_training=True 



CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_35_std_factor_1.yaml model=trans_resnet_3D_34.yaml \
+seed=100 trainer.max_epochs=100 \
current_fold=3 \
+enable_logger=False \
hydra.job.chdir=False \
datamodule.cfg.batch_size=16 \
datamodule.cfg.samples_per_patient=15 \
model.cfg.features_to_include=[0,1,0] \
name=midlrebuttal_contranet-m-010 \
model.cfg.enable_transformer=True \
hydra.job.name=midlrebuttal_contranet-m-010 \
+onlyEval=False test_after_training=True 



CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_35_std_factor_1.yaml model=trans_resnet_3D_34.yaml \
+seed=100 trainer.max_epochs=100 \
current_fold=4 \
+enable_logger=False \
hydra.job.chdir=False \
datamodule.cfg.batch_size=16 \
datamodule.cfg.samples_per_patient=15 \
name=midlrebuttal_contranet-m-010 \
model.cfg.features_to_include=[0,1,0] \
model.cfg.enable_transformer=True \
hydra.job.name=midlrebuttal_contranet-m-010 \
+onlyEval=False test_after_training=True 



CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_35_std_factor_1.yaml model=trans_resnet_3D_34.yaml \
+seed=100 trainer.max_epochs=100 \
current_fold=5 \
+enable_logger=False \
hydra.job.chdir=False \
datamodule.cfg.batch_size=16 \
datamodule.cfg.samples_per_patient=15 \
name=midlrebuttal_contranet-m-010 \
model.cfg.enable_transformer=True \
model.cfg.features_to_include=[0,1,0] \
hydra.job.name=midlrebuttal_contranet-m-010 \
+onlyEval=False test_after_training=True 




CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_35_std_factor_1.yaml model=trans_resnet_3D_34.yaml \
+seed=100 trainer.max_epochs=100 \
current_fold=6 \
+enable_logger=False \
hydra.job.chdir=False \
datamodule.cfg.batch_size=16 \
datamodule.cfg.samples_per_patient=15 \
name=midlrebuttal_contranet-m-010 \
model.cfg.enable_transformer=True \
model.cfg.features_to_include=[0,1,0] \
hydra.job.name=midlrebuttal_contranet-m-010 \
+onlyEval=False test_after_training=True 



CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_35_std_factor_1.yaml model=trans_resnet_3D_34.yaml \
+seed=100 trainer.max_epochs=100 \
current_fold=7 \
+enable_logger=False \
hydra.job.chdir=False \
datamodule.cfg.batch_size=16 \
datamodule.cfg.samples_per_patient=15 \
name=midlrebuttal_contranet-m-010 \
model.cfg.features_to_include=[0,1,0] \
model.cfg.enable_transformer=True \
hydra.job.name=midlrebuttal_contranet-m-010 \
+onlyEval=False test_after_training=True 



CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_35_std_factor_1.yaml model=trans_resnet_3D_34.yaml \
+seed=100 trainer.max_epochs=100 \
current_fold=8 \
+enable_logger=False \
hydra.job.chdir=False \
datamodule.cfg.batch_size=16 \
datamodule.cfg.samples_per_patient=15 \
name=midlrebuttal_contranet-m-010 \
model.cfg.enable_transformer=True \
model.cfg.features_to_include=[0,1,0] \
hydra.job.name=midlrebuttal_contranet-m-010 \
+onlyEval=False test_after_training=True 


CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_35_std_factor_1.yaml model=trans_resnet_3D_34.yaml \
+seed=100 trainer.max_epochs=100 \
current_fold=9 \
+enable_logger=False \
hydra.job.chdir=False \
datamodule.cfg.batch_size=16 \
datamodule.cfg.samples_per_patient=15 \
name=midlrebuttal_contranet-m-010 \
model.cfg.enable_transformer=True \
model.cfg.features_to_include=[0,1,0] \
hydra.job.name=midlrebuttal_contranet-m-010 \
+onlyEval=False test_after_training=True 



CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_35_std_factor_1.yaml model=trans_resnet_3D_34.yaml \
+seed=100 trainer.max_epochs=100 \
current_fold=10 \
+enable_logger=False \
hydra.job.chdir=False \
datamodule.cfg.batch_size=16 \
datamodule.cfg.samples_per_patient=15 \
name=midlrebuttal_contranet-m-010 \
model.cfg.features_to_include=[0,1,0] \
model.cfg.enable_transformer=True \
hydra.job.name=midlrebuttal_contranet-m-010 \
+onlyEval=False test_after_training=True 