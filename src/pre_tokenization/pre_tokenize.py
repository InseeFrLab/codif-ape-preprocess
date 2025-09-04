import hydra
from omegaconf import OmegaConf

from src.constants.paths import FOLDER, PREFIX
from src.pre_tokenization.split import split
from src.utils.io import update_latest_path, upload_parquet, upload_yaml
from src.utils.logger import get_logger

logger = get_logger(name=__name__)


@hydra.main(version_base=None, config_path="configs", config_name="ngram")
def main(cfg):
    root_save_path = PREFIX + FOLDER

    # Save config file
    cfg_yaml = OmegaConf.to_yaml(cfg)
    save_path_yaml = root_save_path + "artifacts/pre_tokenization/"
    cfg_save_path = save_path_yaml + "artifact/hydra_config.yaml"
    upload_yaml(cfg_yaml, cfg_save_path)
    logger.info(f"💾 Saved config file to {cfg_save_path}")

    df_train, df_val, df_test = split(cfg.revision, test_size=cfg.test_size)

    save_path_split = root_save_path + f"{cfg.revision.lower()}/" + "split/"

    logger.info(f"💾 Saving split datasets to {save_path_split}")
    upload_parquet(df_train, save_path_split + "df_train.parquet")
    upload_parquet(df_val, save_path_split + "df_val.parquet")
    upload_parquet(df_test, save_path_split + "df_test.parquet")

    df_train, df_val, df_test = hydra.utils.call(
        cfg, df_train=df_train, df_val=df_val, df_test=df_test
    )

    save_path_preprocessed = (
        root_save_path
        + f"{cfg.revision.lower()}/"
        + "preprocessed/"
        + f"{cfg.name}/"
        + f"remove_stop_words_{cfg.remove_stop_words}_stem_{cfg.stem}/"
    )
    logger.info(f"💾 Saving pre-tokenized datasets to {save_path_preprocessed}")
    upload_parquet(df_train, save_path_preprocessed + "df_train.parquet")
    upload_parquet(df_val, save_path_preprocessed + "df_val.parquet")
    upload_parquet(df_test, save_path_preprocessed + "df_test.parquet")

    # Update latest path for preocessed data
    update_latest_path(
        cfg.revision.upper(),
        PREFIX + FOLDER + f"{cfg.revision.lower()}/",
    )


if __name__ == "__main__":
    main()
