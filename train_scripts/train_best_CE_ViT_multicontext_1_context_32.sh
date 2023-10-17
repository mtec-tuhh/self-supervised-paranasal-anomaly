


CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_35_std_factor_1.yaml model=multicontext_3D_ViT_context_32.yaml \
+seed=100 trainer.max_epochs=100 current_fold=1 \
+enable_logger=True hydra.job.chdir=False datamodule.cfg.batch_size=16 datamodule.cfg.samples_per_patient=15 \
name=multicontext_3D_ViT_context_32_sp_15 hydra.job.name=multicontext_3D_ViT_context_32_sp_15 \
+onlyEval=False test_after_training=True logger=wandb enable_convnet_model=True \
convnet_model_ckpt_path=/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/MRI_HCHS/code/eur-journal/checkpoints/exptname_crop_size_35_std_1_sampleppatient_15/best_fold-1.ckpt




CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_35_std_factor_1.yaml model=multicontext_3D_ViT_context_32.yaml \
+seed=100 trainer.max_epochs=100 current_fold=2 \
+enable_logger=True hydra.job.chdir=False datamodule.cfg.batch_size=16 datamodule.cfg.samples_per_patient=15 \
name=multicontext_3D_ViT_context_32_sp_15 hydra.job.name=multicontext_3D_ViT_context_32_sp_15 \
+onlyEval=False test_after_training=True logger=wandb enable_convnet_model=True \
convnet_model_ckpt_path=/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/MRI_HCHS/code/eur-journal/checkpoints/exptname_crop_size_35_std_1_sampleppatient_15/best_fold-2.ckpt



CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_35_std_factor_1.yaml model=multicontext_3D_ViT_context_32.yaml \
+seed=100 trainer.max_epochs=100 current_fold=3 \
+enable_logger=True hydra.job.chdir=False datamodule.cfg.batch_size=16 datamodule.cfg.samples_per_patient=15 \
name=multicontext_3D_ViT_context_32_sp_15 hydra.job.name=multicontext_3D_ViT_context_32_sp_15 \
+onlyEval=False test_after_training=True logger=wandb enable_convnet_model=True \
convnet_model_ckpt_path=/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/MRI_HCHS/code/eur-journal/checkpoints/exptname_crop_size_35_std_1_sampleppatient_15/best_fold-3.ckpt




CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_35_std_factor_1.yaml model=multicontext_3D_ViT_context_32.yaml \
+seed=100 trainer.max_epochs=100 current_fold=1 \
+enable_logger=True hydra.job.chdir=False datamodule.cfg.batch_size=16 datamodule.cfg.samples_per_patient=10 \
name=multicontext_3D_ViT_context_32_sp_10 hydra.job.name=multicontext_3D_ViT_context_32_sp_10 \
+onlyEval=False test_after_training=True logger=wandb enable_convnet_model=True \
convnet_model_ckpt_path=/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/MRI_HCHS/code/eur-journal/checkpoints/exptname_crop_size_35_std_1_sampleppatient_10/best_fold-1.ckpt




CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_35_std_factor_1.yaml model=multicontext_3D_ViT_context_32.yaml \
+seed=100 trainer.max_epochs=100 current_fold=2 \
+enable_logger=True hydra.job.chdir=False datamodule.cfg.batch_size=16 datamodule.cfg.samples_per_patient=10 \
name=multicontext_3D_ViT_context_32_sp_10 hydra.job.name=multicontext_3D_ViT_context_32_sp_10 \
+onlyEval=False test_after_training=True logger=wandb enable_convnet_model=True \
convnet_model_ckpt_path=/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/MRI_HCHS/code/eur-journal/checkpoints/exptname_crop_size_35_std_1_sampleppatient_10/best_fold-2.ckpt



CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_35_std_factor_1.yaml model=multicontext_3D_ViT_context_32.yaml \
+seed=100 trainer.max_epochs=100 current_fold=3 \
+enable_logger=True hydra.job.chdir=False datamodule.cfg.batch_size=16 datamodule.cfg.samples_per_patient=10 \
name=multicontext_3D_ViT_context_32_sp_10 hydra.job.name=multicontext_3D_ViT_context_32_sp_10 \
+onlyEval=False test_after_training=True logger=wandb enable_convnet_model=True \
convnet_model_ckpt_path=/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/MRI_HCHS/code/eur-journal/checkpoints/exptname_crop_size_35_std_1_sampleppatient_10/best_fold-3.ckpt


