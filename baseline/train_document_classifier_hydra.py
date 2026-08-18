from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from train_document_classifier import namespace_from_hydra_config, train


@hydra.main(config_path="configs", config_name="doc_roberta", version_base="1.3")
def main(cfg: DictConfig) -> None:
    config_dir = Path(__file__).resolve().parent / "configs"
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    args = namespace_from_hydra_config(cfg_dict, config_dir)
    train(args)


if __name__ == "__main__":
    main()
