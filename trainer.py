import hydra
from hydra.utils import instantiate
from pytorch_lightning.loggers import TensorBoardLogger

from dataloading.osteosarcomaDataModule import OsteosarcomaDataModule
from network_module import Net


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg):
    if cfg.pretrained:
        base_run_name = str(cfg.run_name) + "_pretrained"
    else:
        base_run_name = str(cfg.run_name)
    
    for k in range(cfg.n_folds):
        run_name = f"{cfg.experiment_name}/{base_run_name}/{cfg.datamodule.img_size}"
        if cfg.n_folds > 1:
            run_name += f"/{k}_fold"

        tensorboard_logger = TensorBoardLogger(
            save_dir="logs",
            name=run_name,
        )

        dm = instantiate(cfg.datamodule, k=k)
        dm.prepare_data()

        model = instantiate(cfg.model.object, num_classes=dm.num_classes)
        net = Net(
            model=model,
            criterion=instantiate(cfg.criterion, weight=dm.class_weights),
            num_classes=dm.num_classes,
            optimizer=instantiate(cfg.optimizer, params=model.parameters()),
            scheduler=cfg.scheduler,
        )

        trainer = instantiate(cfg.trainer, logger=tensorboard_logger)
        trainer.fit(net, dm)


if __name__ == "__main__":
    main()
