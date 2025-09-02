import hydra
from omegaconf import OmegaConf

from src.constants.paths import FOLDER, PREFIX, PREPROCESSED_FOLDER
from src.utils.io import upload_parquet, upload_yaml
from src.utils.logger import get_logger

logger = get_logger(name=__name__)


@hydra.main(version_base=None, config_path="configs", config_name="ngram")
def main(cfg):
    # Save config file
    cfg_yaml = OmegaConf.to_yaml(cfg)
    save_path = PREFIX + FOLDER + f"{cfg.revision.lower()}/" + PREPROCESSED_FOLDER
    cfg_save_path = save_path + "artifact/hydra_config.yaml"
    upload_yaml(cfg_yaml, cfg_save_path)
    logger.info(f"💾 Saved config file to {cfg_save_path}")

    df_train, df_val, df_test = hydra.utils.call(cfg)

    logger.info(f"💾 Saving pre-tokenized datasets to {save_path}")
    upload_parquet(df_train, save_path + "df_train.parquet")
    upload_parquet(df_val, save_path + "df_val.parquet")
    upload_parquet(df_test, save_path + "df_test.parquet")


if __name__ == "__main__":
    main()
