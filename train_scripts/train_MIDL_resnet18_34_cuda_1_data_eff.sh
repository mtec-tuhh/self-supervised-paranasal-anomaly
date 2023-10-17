




#Resnet  18


CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_35_std_factor_1.yaml model=resnet_3D_18.yaml \
+seed=100 trainer.max_epochs=100 \
current_fold=1 \
+enable_logger=False \
hydra.job.chdir=False \
datamodule.cfg.batch_size=16 \
datamodule.cfg.samples_per_patient=1 \
name=midlrebuttal_dataeff_resnet18 \
hydra.job.name=midlrebuttal_dataeff_resnet18 \
+onlyEval=False test_after_training=True

CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_35_std_factor_1.yaml model=resnet_3D_18.yaml \
+seed=100 trainer.max_epochs=100 \
current_fold=2 \
+enable_logger=False \
hydra.job.chdir=False \
datamodule.cfg.batch_size=16 \
datamodule.cfg.samples_per_patient=1 \
name=midlrebuttal_dataeff_resnet18 \
hydra.job.name=midlrebuttal_dataeff_resnet18 \
+onlyEval=False test_after_training=True


CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_35_std_factor_1.yaml model=resnet_3D_18.yaml \
+seed=100 trainer.max_epochs=100 \
current_fold=3 \
+enable_logger=False \
hydra.job.chdir=False \
datamodule.cfg.batch_size=16 \
datamodule.cfg.samples_per_patient=1 \
name=midlrebuttal_dataeff_resnet18 \
hydra.job.name=midlrebuttal_dataeff_resnet18 \
+onlyEval=False test_after_training=True




CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_35_std_factor_1.yaml model=resnet_3D_18.yaml \
+seed=100 trainer.max_epochs=100 \
current_fold=4 \
+enable_logger=False \
hydra.job.chdir=False \
datamodule.cfg.batch_size=16 \
datamodule.cfg.samples_per_patient=1 \
name=midlrebuttal_dataeff_resnet18 \
hydra.job.name=midlrebuttal_dataeff_resnet18 \
+onlyEval=False test_after_training=True

CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_35_std_factor_1.yaml model=resnet_3D_18.yaml \
+seed=100 trainer.max_epochs=100 \
current_fold=5 \
+enable_logger=False \
hydra.job.chdir=False \
datamodule.cfg.batch_size=16 \
datamodule.cfg.samples_per_patient=1 \
name=midlrebuttal_dataeff_resnet18 \
hydra.job.name=midlrebuttal_dataeff_resnet18 \
+onlyEval=False test_after_training=True


CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_35_std_factor_1.yaml model=resnet_3D_18.yaml \
+seed=100 trainer.max_epochs=100 \
current_fold=6 \
+enable_logger=False \
hydra.job.chdir=False \
datamodule.cfg.batch_size=16 \
datamodule.cfg.samples_per_patient=1 \
name=midlrebuttal_dataeff_resnet18 \
hydra.job.name=midlrebuttal_dataeff_resnet18 \
+onlyEval=False test_after_training=True



CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_35_std_factor_1.yaml model=resnet_3D_18.yaml \
+seed=100 trainer.max_epochs=100 \
current_fold=7 \
+enable_logger=False \
hydra.job.chdir=False \
datamodule.cfg.batch_size=16 \
datamodule.cfg.samples_per_patient=1 \
name=midlrebuttal_dataeff_resnet18 \
hydra.job.name=midlrebuttal_dataeff_resnet18 \
+onlyEval=False test_after_training=True

CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_35_std_factor_1.yaml model=resnet_3D_18.yaml \
+seed=100 trainer.max_epochs=100 \
current_fold=8 \
+enable_logger=False \
hydra.job.chdir=False \
datamodule.cfg.batch_size=16 \
datamodule.cfg.samples_per_patient=1 \
name=midlrebuttal_dataeff_resnet18 \
hydra.job.name=midlrebuttal_dataeff_resnet18 \
+onlyEval=False test_after_training=True


CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_35_std_factor_1.yaml model=resnet_3D_18.yaml \
+seed=100 trainer.max_epochs=100 \
current_fold=9 \
+enable_logger=False \
hydra.job.chdir=False \
datamodule.cfg.batch_size=16 \
datamodule.cfg.samples_per_patient=1 \
name=midlrebuttal_dataeff_resnet18 \
hydra.job.name=midlrebuttal_dataeff_resnet18 \
+onlyEval=False test_after_training=True



CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_35_std_factor_1.yaml model=resnet_3D_18.yaml \
+seed=100 trainer.max_epochs=100 \
current_fold=10 \
+enable_logger=False \
hydra.job.chdir=False \
datamodule.cfg.batch_size=16 \
datamodule.cfg.samples_per_patient=1 \
name=midlrebuttal_dataeff_resnet18 \
hydra.job.name=midlrebuttal_dataeff_resnet18 \
+onlyEval=False test_after_training=True





#ResNet 34

CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_35_std_factor_1.yaml model=resnet_3D_34.yaml \
+seed=100 trainer.max_epochs=100 \
current_fold=1 \
+enable_logger=False \
hydra.job.chdir=False \
datamodule.cfg.batch_size=16 \
datamodule.cfg.samples_per_patient=1 \
name=midlrebuttal_dataeff_resnet34 \
hydra.job.name=midlrebuttal_dataeff_resnet34 \
+onlyEval=False test_after_training=True


CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_35_std_factor_1.yaml model=resnet_3D_34.yaml \
+seed=100 trainer.max_epochs=100 \
current_fold=2 \
+enable_logger=False \
hydra.job.chdir=False \
datamodule.cfg.batch_size=16 \
datamodule.cfg.samples_per_patient=1 \
name=midlrebuttal_dataeff_resnet34 \
hydra.job.name=midlrebuttal_dataeff_resnet34 \
+onlyEval=False test_after_training=True


CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_35_std_factor_1.yaml model=resnet_3D_34.yaml \
+seed=100 trainer.max_epochs=100 \
current_fold=3 \
+enable_logger=False \
hydra.job.chdir=False \
datamodule.cfg.batch_size=16 \
datamodule.cfg.samples_per_patient=1 \
name=midlrebuttal_dataeff_resnet34 \
hydra.job.name=midlrebuttal_dataeff_resnet34 \
+onlyEval=False test_after_training=True


CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_35_std_factor_1.yaml model=resnet_3D_34.yaml \
+seed=100 trainer.max_epochs=100 \
current_fold=4 \
+enable_logger=False \
hydra.job.chdir=False \
datamodule.cfg.batch_size=16 \
datamodule.cfg.samples_per_patient=1 \
name=midlrebuttal_dataeff_resnet34 \
hydra.job.name=midlrebuttal_dataeff_resnet34 \
+onlyEval=False test_after_training=True


CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_35_std_factor_1.yaml model=resnet_3D_34.yaml \
+seed=100 trainer.max_epochs=100 \
current_fold=5 \
+enable_logger=False \
hydra.job.chdir=False \
datamodule.cfg.batch_size=16 \
datamodule.cfg.samples_per_patient=1 \
name=midlrebuttal_dataeff_resnet34 \
hydra.job.name=midlrebuttal_dataeff_resnet34 \
+onlyEval=False test_after_training=True


CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_35_std_factor_1.yaml model=resnet_3D_34.yaml \
+seed=100 trainer.max_epochs=100 \
current_fold=6 \
+enable_logger=False \
hydra.job.chdir=False \
datamodule.cfg.batch_size=16 \
datamodule.cfg.samples_per_patient=1 \
name=midlrebuttal_dataeff_resnet34 \
hydra.job.name=midlrebuttal_dataeff_resnet34 \
+onlyEval=False test_after_training=True


CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_35_std_factor_1.yaml model=resnet_3D_34.yaml \
+seed=100 trainer.max_epochs=100 \
current_fold=7 \
+enable_logger=False \
hydra.job.chdir=False \
datamodule.cfg.batch_size=16 \
datamodule.cfg.samples_per_patient=1 \
name=midlrebuttal_dataeff_resnet34 \
hydra.job.name=midlrebuttal_dataeff_resnet34 \
+onlyEval=False test_after_training=True



CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_35_std_factor_1.yaml model=resnet_3D_34.yaml \
+seed=100 trainer.max_epochs=100 \
current_fold=8 \
+enable_logger=False \
hydra.job.chdir=False \
datamodule.cfg.batch_size=16 \
datamodule.cfg.samples_per_patient=1 \
name=midlrebuttal_dataeff_resnet34 \
hydra.job.name=midlrebuttal_dataeff_resnet34 \
+onlyEval=False test_after_training=True



CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_35_std_factor_1.yaml model=resnet_3D_34.yaml \
+seed=100 trainer.max_epochs=100 \
current_fold=9 \
+enable_logger=False \
hydra.job.chdir=False \
datamodule.cfg.batch_size=16 \
datamodule.cfg.samples_per_patient=1 \
name=midlrebuttal_dataeff_resnet34 \
hydra.job.name=midlrebuttal_dataeff_resnet34 \
+onlyEval=False test_after_training=True



CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_35_std_factor_1.yaml model=resnet_3D_34.yaml \
+seed=100 trainer.max_epochs=100 \
current_fold=10 \
+enable_logger=False \
hydra.job.chdir=False \
datamodule.cfg.batch_size=16 \
datamodule.cfg.samples_per_patient=1 \
name=midlrebuttal_dataeff_resnet34 \
hydra.job.name=midlrebuttal_dataeff_resnet34 \
+onlyEval=False test_after_training=True

