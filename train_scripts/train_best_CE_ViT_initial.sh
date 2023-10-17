


CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python33 run.py  \
datamodule=crop_size_35_std_factor_1.yaml model=ViT_3D_bs_16_ps_4_dim_8_d_2_h_8_mlp_16.yaml \
+seed=100 trainer.max_epochs=100 current_fold=1 \
+enable_logger=True hydra.job.chdir=False datamodule.cfg.batch_size=16 datamodule.cfg.samples_per_patient=15 \
name=ViT_3D_bs_16_ps_4_dim_8_d_2_h_8_mlp_16 hydra.job.name=ViT_3D_bs_16_ps_4_dim_8_d_2_h_8_mlp_16 \
+onlyEval=False test_after_training=True logger=wandb


CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_35_std_factor_1.yaml model=ViT_3D_bs_16_ps_4_dim_8_d_4_h_8_mlp_16.yaml \
+seed=100 trainer.max_epochs=100 current_fold=2 \
+enable_logger=True hydra.job.chdir=False datamodule.cfg.batch_size=16 datamodule.cfg.samples_per_patient=15 \
name=ViT_3D_bs_16_ps_4_dim_8_d_4_h_8_mlp_16 hydra.job.name=ViT_3D_bs_16_ps_4_dim_8_d_4_h_8_mlp_16 \
+onlyEval=False test_after_training=True logger=wandb

CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_35_std_factor_1.yaml model=ViT_3D_bs_16_ps_4_dim_16_d_2_h_8_mlp_32.yaml \
+seed=100 trainer.max_epochs=100 current_fold=3 \
+enable_logger=True hydra.job.chdir=False datamodule.cfg.batch_size=16 datamodule.cfg.samples_per_patient=15 \
name=ViT_3D_bs_16_ps_4_dim_16_d_2_h_8_mlp_32 hydra.job.name=ViT_3D_bs_16_ps_4_dim_16_d_2_h_8_mlp_32 \
+onlyEval=False test_after_training=True logger=wandb

CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_35_std_factor_1.yaml model=ViT_3D_bs_16_ps_4_dim_16_d_4_h_8_mlp_32.yaml \
+seed=100 trainer.max_epochs=100 current_fold=4 \
+enable_logger=True hydra.job.chdir=False datamodule.cfg.batch_size=16 datamodule.cfg.samples_per_patient=15 \
name=ViT_3D_bs_16_ps_4_dim_16_d_4_h_8_mlp_32 hydra.job.name=ViT_3D_bs_16_ps_4_dim_16_d_4_h_8_mlp_32 \
+onlyEval=False test_after_training=True logger=wandb





CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python33 run.py  \
datamodule=crop_size_35_std_factor_1.yaml model=ViT_3D_bs_16_ps_8_dim_8_d_2_h_8_mlp_16.yaml \
+seed=100 trainer.max_epochs=100 current_fold=1 \
+enable_logger=True hydra.job.chdir=False datamodule.cfg.batch_size=16 datamodule.cfg.samples_per_patient=15 \
name=ViT_3D_bs_16_ps_8_dim_8_d_2_h_8_mlp_16 hydra.job.name=ViT_3D_bs_16_ps_8_dim_8_d_2_h_8_mlp_16 \
+onlyEval=False test_after_training=True logger=wandb


CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_35_std_factor_1.yaml model=ViT_3D_bs_16_ps_8_dim_8_d_4_h_8_mlp_16_3D.yaml \
+seed=100 trainer.max_epochs=100 current_fold=2 \
+enable_logger=True hydra.job.chdir=False datamodule.cfg.batch_size=16 datamodule.cfg.samples_per_patient=15 \
name=ViT_3D_bs_16_ps_8_dim_8_d_4_h_8_mlp_16_3D hydra.job.name=ViT_3D_bs_16_ps_8_dim_8_d_4_h_8_mlp_16_3D \
+onlyEval=False test_after_training=True logger=wandb

CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_35_std_factor_1.yaml model=ViT_3D_bs_16_ps_8_dim_16_d_2_h_8_mlp_32.yaml \
+seed=100 trainer.max_epochs=100 current_fold=3 \
+enable_logger=True hydra.job.chdir=False datamodule.cfg.batch_size=16 datamodule.cfg.samples_per_patient=15 \
name=ViT_3D_bs_16_ps_8_dim_16_d_2_h_8_mlp_32 hydra.job.name=ViT_3D_bs_16_ps_8_dim_16_d_2_h_8_mlp_32 \
+onlyEval=False test_after_training=True logger=wandb

CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_35_std_factor_1.yaml model=ViT_3D_bs_16_ps_8_dim_16_d_4_h_8_mlp_32.yaml \
+seed=100 trainer.max_epochs=100 current_fold=4 \
+enable_logger=True hydra.job.chdir=False datamodule.cfg.batch_size=16 datamodule.cfg.samples_per_patient=15 \
name=ViT_3D_bs_16_ps_8_dim_16_d_4_h_8_mlp_32 hydra.job.name=ViT_3D_bs_16_ps_8_dim_16_d_4_h_8_mlp_32 \
+onlyEval=False test_after_training=True logger=wandb
