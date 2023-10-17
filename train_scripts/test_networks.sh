

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
datamodule=crop_size_35_std_factor_1_JAMA.yaml model=resnet_3D_18.yaml \
+seed=100 trainer.max_epochs=100 \
total_folds=5 \
current_fold=1 \
+enable_logger=False \
hydra.job.chdir=False \
datamodule.cfg.batch_size=1 \
datamodule.cfg.samples_per_patient=1 \
name=ResNet18_cc_35_test \
hydra.job.name=ResNet18_cc_35_test \
+load_checkpoint=ResNet18_cc_35 \
+onlyEval=False test_after_training=True


#Resnet  18


CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_35_std_factor_1.yaml model=resnet_3D_18.yaml \
+seed=100 trainer.max_epochs=100 \
current_fold=1 \
+enable_logger=False \
hydra.job.chdir=False \
datamodule.cfg.batch_size=16 \
datamodule.cfg.samples_per_patient=15 \
name=motion_degrees_40_resnet18 \
hydra.job.name=motion_degrees_40_resnet18 \
+load_checkpoint=exptname_resnet_18 \
+onlyEval=False test_after_training=True

CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_35_std_factor_1.yaml model=resnet_3D_18.yaml \
+seed=100 trainer.max_epochs=100 \
current_fold=2 \
+enable_logger=False \
hydra.job.chdir=False \
datamodule.cfg.batch_size=16 \
datamodule.cfg.samples_per_patient=15 \
name=motion_degrees_40_resnet18 \
hydra.job.name=motion_degrees_40_resnet18 \
+load_checkpoint=exptname_resnet_18 \
+onlyEval=False test_after_training=True


CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_35_std_factor_1.yaml model=resnet_3D_18.yaml \
+seed=100 trainer.max_epochs=100 \
current_fold=3 \
+enable_logger=False \
hydra.job.chdir=False \
datamodule.cfg.batch_size=16 \
datamodule.cfg.samples_per_patient=15 \
name=motion_degrees_40_resnet18 \
hydra.job.name=motion_degrees_40_resnet18 \
+load_checkpoint=exptname_resnet_18 \
+onlyEval=False test_after_training=True



#ResNet 34

CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_35_std_factor_1.yaml model=resnet_3D_34.yaml \
+seed=100 trainer.max_epochs=100 \
current_fold=1 \
+enable_logger=False \
hydra.job.chdir=False \
datamodule.cfg.batch_size=16 \
datamodule.cfg.samples_per_patient=15 \
name=motion_degrees_40_resnet34 \
hydra.job.name=motion_degrees_40_resnet34 \
+load_checkpoint=exptname_resnet_34 \
+onlyEval=False test_after_training=True


CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_35_std_factor_1.yaml model=resnet_3D_34.yaml \
+seed=100 trainer.max_epochs=100 \
current_fold=2 \
+enable_logger=False \
hydra.job.chdir=False \
datamodule.cfg.batch_size=16 \
datamodule.cfg.samples_per_patient=15 \
name=motion_degrees_40_resnet34 \
hydra.job.name=motion_degrees_40_resnet34 \
+load_checkpoint=exptname_resnet_34 \
+onlyEval=False test_after_training=True


CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_35_std_factor_1.yaml model=resnet_3D_34.yaml \
+seed=100 trainer.max_epochs=100 \
current_fold=3 \
+enable_logger=False \
hydra.job.chdir=False \
datamodule.cfg.batch_size=16 \
datamodule.cfg.samples_per_patient=15 \
name=motion_degrees_40_resnet34 \
hydra.job.name=motion_degrees_40_resnet34 \
+load_checkpoint=exptname_resnet_34 \
+onlyEval=False test_after_training=True




#ResNet 50

CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_35_std_factor_1.yaml model=resnet_3D_50.yaml \
+seed=100 trainer.max_epochs=100 \
current_fold=1 \
+enable_logger=False \
hydra.job.chdir=False \
datamodule.cfg.batch_size=16 \
datamodule.cfg.samples_per_patient=15 \
name=motion_degrees_40_resnet50 \
hydra.job.name=motion_degrees_40_resnet50 \
+load_checkpoint=exptname_resnet_50 \
+onlyEval=False test_after_training=True

CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_35_std_factor_1.yaml model=resnet_3D_50.yaml \
+seed=100 trainer.max_epochs=100 \
current_fold=2 \
+enable_logger=False \
hydra.job.chdir=False \
datamodule.cfg.batch_size=16 \
datamodule.cfg.samples_per_patient=15 \
name=motion_degrees_40_resnet50 \
hydra.job.name=motion_degrees_40_resnet50 \
+load_checkpoint=exptname_resnet_50 \
+onlyEval=False test_after_training=True


CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_35_std_factor_1.yaml model=resnet_3D_50.yaml \
+seed=100 trainer.max_epochs=100 \
current_fold=3 \
+enable_logger=False \
hydra.job.chdir=False \
datamodule.cfg.batch_size=16 \
datamodule.cfg.samples_per_patient=15 \
name=motion_degrees_40_resnet50 \
hydra.job.name=motion_degrees_40_resnet50 \
+load_checkpoint=exptname_resnet_50 \
+onlyEval=False test_after_training=True

#3d vit

CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_35_std_factor_1.yaml model=ViT_3D.yaml \
+seed=100 trainer.max_epochs=100 \
current_fold=1 \
+enable_logger=False \
hydra.job.chdir=False \
datamodule.cfg.batch_size=16 \
datamodule.cfg.samples_per_patient=1 \
name=motion_degrees_40_ViT \
hydra.job.name=motion_degrees_40_ViT \
+load_checkpoint=exptname_vit_d_2_h_4_emb_512 \
+onlyEval=False test_after_training=True 


CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_35_std_factor_1.yaml model=ViT_3D.yaml \
+seed=100 trainer.max_epochs=100 \
current_fold=2 \
+enable_logger=False \
hydra.job.chdir=False \
datamodule.cfg.batch_size=16 \
datamodule.cfg.samples_per_patient=1 \
name=motion_degrees_40_ViT \
hydra.job.name=motion_degrees_40_ViT \
+load_checkpoint=exptname_vit_d_2_h_4_emb_512 \
+onlyEval=False test_after_training=True 


CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_35_std_factor_1.yaml model=ViT_3D.yaml \
+seed=100 trainer.max_epochs=100 \
current_fold=3 \
+enable_logger=False \
hydra.job.chdir=False \
datamodule.cfg.batch_size=16 \
datamodule.cfg.samples_per_patient=1 \
name=motion_degrees_40_ViT \
hydra.job.name=motion_degrees_40_ViT \
+load_checkpoint=exptname_vit_d_2_h_4_emb_512 \
+onlyEval=False test_after_training=True 


#ContraNet 

CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_35_std_factor_1.yaml model=trans_resnet_3D_18.yaml \
+seed=100 trainer.max_epochs=100 \
current_fold=1 \
+enable_logger=False \
hydra.job.chdir=False \
datamodule.cfg.batch_size=16 \
datamodule.cfg.samples_per_patient=15 \
name=motion_degrees_40_contranet-s-2 \
model.cfg.enable_transformer=True \
hydra.job.name=motion_degrees_40_contranet-s-2 \
+load_checkpoint=exptname_trans_resnet_18_d_2_h_4_emb_512_clstoken \
+onlyEval=False test_after_training=True 


CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_35_std_factor_1.yaml model=trans_resnet_3D_18.yaml \
+seed=100 trainer.max_epochs=100 \
current_fold=2 \
+enable_logger=False \
hydra.job.chdir=False \
datamodule.cfg.batch_size=16 \
datamodule.cfg.samples_per_patient=15 \
name=motion_degrees_40_contranet-s-2 \
model.cfg.enable_transformer=True \
hydra.job.name=motion_degrees_40_contranet-s-2 \
+load_checkpoint=exptname_trans_resnet_18_d_2_h_4_emb_512_clstoken \
+onlyEval=False test_after_training=True 



CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_35_std_factor_1.yaml model=trans_resnet_3D_18.yaml \
+seed=100 trainer.max_epochs=100 \
current_fold=3 \
+enable_logger=False \
hydra.job.chdir=False \
datamodule.cfg.batch_size=16 \
datamodule.cfg.samples_per_patient=15 \
name=motion_degrees_40_contranet-s-2 \
model.cfg.enable_transformer=True \
hydra.job.name=motion_degrees_40_contranet-s-2 \
+load_checkpoint=exptname_trans_resnet_18_d_2_h_4_emb_512_clstoken \
+onlyEval=False test_after_training=True 