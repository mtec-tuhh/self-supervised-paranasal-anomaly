CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python run.py  \
datamodule=crop_size_35_std_factor_1_16.yaml model=resnet_3D_18.yaml \
+seed=100 trainer.max_epochs=100 current_fold=1 \
+enable_logger=False hydra.job.chdir=False hydra.job.name=aug_batch_size_16 datamodule.cfg.batch_size=16 name=aug_batch_size_16 \
+onlyEval=False test_after_training=True 


CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python run.py  \
datamodule=crop_size_35_std_factor_1_32.yaml model=resnet_3D_18.yaml \
+seed=100 trainer.max_epochs=100 current_fold=1 \
+enable_logger=False hydra.job.chdir=False hydra.job.name=batch_size_32 datamodule.cfg.batch_size=32 name=batch_size_32 \
+onlyEval=False test_after_training=True 


CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python run.py  \
datamodule=crop_size_35_std_factor_1_64.yaml model=resnet_3D_18.yaml \
+seed=100 trainer.max_epochs=100 current_fold=1 \
+enable_logger=False hydra.job.chdir=False hydra.job.name=batch_size_64 datamodule.cfg.batch_size=64 name=batch_size_64 \
+onlyEval=False test_after_training=True 


CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python run.py  \
datamodule=crop_size_35_std_factor_1_128.yaml model=resnet_3D_18.yaml \
+seed=100 trainer.max_epochs=100 current_fold=1 \
+enable_logger=False hydra.job.chdir=False hydra.job.name=batch_size_128 datamodule.cfg.batch_size=128 name=batch_size_128 \
+onlyEval=False test_after_training=True 


CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python run.py  \
datamodule=crop_size_35_std_factor_1_256.yaml model=resnet_3D_18.yaml \
+seed=100 trainer.max_epochs=100 current_fold=1 \
+enable_logger=False hydra.job.chdir=False hydra.job.name=batch_size_256 datamodule.cfg.batch_size=256 name=batch_size_256 \
+onlyEval=False test_after_training=True 



