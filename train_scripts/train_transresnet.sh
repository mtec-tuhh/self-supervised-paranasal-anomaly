

CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_35_std_factor_1.yaml model=trans_resnet_3D_18.yaml \
+seed=100 trainer.max_epochs=100 \
current_fold=2 \
+enable_logger=False \
hydra.job.chdir=False \
datamodule.cfg.batch_size=16 \
datamodule.cfg.samples_per_patient=15 \
model.cfg.t_depth=2 \
model.cfg.t_heads=4 \
model.cfg.embedding_dim=512 \
name=trans_resnet_18_d_2_h_4_emb_512_clstoken \
hydra.job.name=trans_resnet_18_d_2_h_4_emb_512_clstoken \
+onlyEval=False test_after_training=True 



CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python3 run.py  \
datamodule=crop_size_35_std_factor_1.yaml model=trans_resnet_3D_18.yaml \
+seed=100 trainer.max_epochs=100 \
current_fold=3 \
+enable_logger=False \
hydra.job.chdir=False \
datamodule.cfg.batch_size=16 \
datamodule.cfg.samples_per_patient=15 \
model.cfg.t_depth=2 \
model.cfg.t_heads=4 \
model.cfg.embedding_dim=512 \
name=trans_resnet_18_d_2_h_4_emb_512_clstoken \
hydra.job.name=trans_resnet_18_d_2_h_4_emb_512_clstoken \
+onlyEval=False test_after_training=True 