import hydra


@hydra.main(version_base=None, config_path="configs", config_name="ngram")
def main(cfg):
    df_train, df_val, df_test = hydra.utils.call(cfg)


if __name__ == "__main__":
    main()
