HYDRA_FULL_ERROR=1 python run.py data_dir=datasets_SPIE_small model=ae_3D.yaml +seed=100 trainer.max_epochs=100 +num_folds=3 +enable_logger=False hydra.job.chdir=False hydra.job.name=uad_AE_LV_128_BS_16_DS_small model.cfg.latent_size=128 datamodule.cfg.batch_size=16 name=AELatentVectorExpBS16DSSmall +onlyEval=False
HYDRA_FULL_ERROR=1 python run.py data_dir=datasets_SPIE_small model=ae_3D.yaml +seed=100 trainer.max_epochs=100 +num_folds=3 +enable_logger=False hydra.job.chdir=False hydra.job.name=uad_AE_LV_256_BS_16_DS_small model.cfg.latent_size=256 datamodule.cfg.batch_size=16 name=AELatentVectorExpBS16DSSmall +onlyEval=False
HYDRA_FULL_ERROR=1 python run.py data_dir=datasets_SPIE_small model=ae_3D.yaml +seed=100 trainer.max_epochs=100 +num_folds=3 +enable_logger=False hydra.job.chdir=False hydra.job.name=uad_AE_LV_512_BS_16_DS_small model.cfg.latent_size=512 datamodule.cfg.batch_size=16 name=AELatentVectorExpBS16DSSmall +onlyEval=False  
HYDRA_FULL_ERROR=1 python run.py data_dir=datasets_SPIE_small model=ae_3D.yaml +seed=100 trainer.max_epochs=100 +num_folds=3 +enable_logger=False hydra.job.chdir=False hydra.job.name=uad_AE_LV_1024_BS_16_DS_small model.cfg.latent_size=1024 datamodule.cfg.batch_size=16 name=AELatentVectorExpBS16DSSmall +onlyEval=False 


HYDRA_FULL_ERROR=1 python run.py data_dir=datasets_SPIE_medium model=ae_3D.yaml +seed=100 trainer.max_epochs=100 +num_folds=3 +enable_logger=False hydra.job.chdir=False hydra.job.name=uad_AE_LV_128_BS_16_DS_medium model.cfg.latent_size=128 datamodule.cfg.batch_size=16 name=AELatentVectorExpBS16DSMedium +onlyEval=False
HYDRA_FULL_ERROR=1 python run.py data_dir=datasets_SPIE_medium model=ae_3D.yaml +seed=100 trainer.max_epochs=100 +num_folds=3 +enable_logger=False hydra.job.chdir=False hydra.job.name=uad_AE_LV_256_BS_16_DS_medium model.cfg.latent_size=256 datamodule.cfg.batch_size=16 name=AELatentVectorExpBS16DSMedium +onlyEval=False
HYDRA_FULL_ERROR=1 python run.py data_dir=datasets_SPIE_medium model=ae_3D.yaml +seed=100 trainer.max_epochs=100 +num_folds=3 +enable_logger=False hydra.job.chdir=False hydra.job.name=uad_AE_LV_512_BS_16_DS_medium model.cfg.latent_size=512 datamodule.cfg.batch_size=16 name=AELatentVectorExpBS16DSMedium +onlyEval=False  
HYDRA_FULL_ERROR=1 python run.py data_dir=datasets_SPIE_medium model=ae_3D.yaml +seed=100 trainer.max_epochs=100 +num_folds=3 +enable_logger=False hydra.job.chdir=False hydra.job.name=uad_AE_LV_1024_BS_16_DS_medium model.cfg.latent_size=1024 datamodule.cfg.batch_size=16 name=AELatentVectorExpBS16DSMedium +onlyEval=False 



HYDRA_FULL_ERROR=1 python run.py data_dir=datasets_SPIE_large model=ae_3D.yaml +seed=100 trainer.max_epochs=100 +num_folds=3 +enable_logger=False hydra.job.chdir=False hydra.job.name=uad_AE_LV_128_BS_16_DS_large model.cfg.latent_size=128 datamodule.cfg.batch_size=16 name=AELatentVectorExpBS16DSLarge +onlyEval=False
HYDRA_FULL_ERROR=1 python run.py data_dir=datasets_SPIE_large model=ae_3D.yaml +seed=100 trainer.max_epochs=100 +num_folds=3 +enable_logger=False hydra.job.chdir=False hydra.job.name=uad_AE_LV_256_BS_16_DS_large model.cfg.latent_size=256 datamodule.cfg.batch_size=16 name=AELatentVectorExpBS16DSLarge +onlyEval=False
HYDRA_FULL_ERROR=1 python run.py data_dir=datasets_SPIE_large model=ae_3D.yaml +seed=100 trainer.max_epochs=100 +num_folds=3 +enable_logger=False hydra.job.chdir=False hydra.job.name=uad_AE_LV_512_BS_16_DS_large model.cfg.latent_size=512 datamodule.cfg.batch_size=16 name=AELatentVectorExpBS16DSLarge +onlyEval=False  
HYDRA_FULL_ERROR=1 python run.py data_dir=datasets_SPIE_large model=ae_3D.yaml +seed=100 trainer.max_epochs=100 +num_folds=3 +enable_logger=False hydra.job.chdir=False hydra.job.name=uad_AE_LV_1024_BS_16_DS_large model.cfg.latent_size=1024 datamodule.cfg.batch_size=16 name=AELatentVectorExpBS16DSLarge +onlyEval=False 


HYDRA_FULL_ERROR=1 python run.py +show_result=True +csv_file_names=[uad_AE_LV_128_BS_16_DS_small,uad_AE_LV_256_BS_16_DS_small,uad_AE_LV_512_BS_16_DS_small,uad_AE_LV_1024_BS_16_DS_small] name=AELatentVectorExpBS16DSSmall
HYDRA_FULL_ERROR=1 python run.py +show_result=True +csv_file_names=[uad_AE_LV_128_BS_16_DS_medium,uad_AE_LV_256_BS_16_DS_medium,uad_AE_LV_512_BS_16_DS_medium,uad_AE_LV_1024_BS_16_DS_medium] name=AELatentVectorExpBS16DSMedium
HYDRA_FULL_ERROR=1 python run.py +show_result=True +csv_file_names=[uad_AE_LV_128_BS_16_DS_large,uad_AE_LV_256_BS_16_DS_large,uad_AE_LV_512_BS_16_DS_large,uad_AE_LV_1024_BS_16_DS_large] name=AELatentVectorExpBS16DSLarge
