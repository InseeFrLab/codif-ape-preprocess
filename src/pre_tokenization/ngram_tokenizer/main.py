from src.constants import (
    CATEGORICAL_FEATURES,
    NACE_REV2_1_COLUMN,
    NACE_REV2_COLUMN,
    TEXT_FEATURE,
)
from src.pre_tokenization.ngram_tokenizer.utils import (
    clean_categorical_features,
    clean_text_feature,
)
from src.utils.logger import get_logger

logger = get_logger(name=__name__)


def ngram_pretokenize(
    revision, df_train, df_val, df_test, remove_stop_words=True, stem=True, **kwargs
):
    def clean_df(df):
        df[TEXT_FEATURE] = clean_text_feature(
            df[TEXT_FEATURE], remove_stop_words=remove_stop_words, stem=stem
        )
        df = clean_categorical_features(
            df, categorical_features=CATEGORICAL_FEATURES, y=Y
        )
        return df

    Y = NACE_REV2_COLUMN if revision == "NAF2008" else NACE_REV2_1_COLUMN

    df_train = clean_df(df_train)
    df_val = clean_df(df_val)
    df_test = clean_df(df_test)

    # # Adding the true labels to the training set
    logger.info("✅ Data Processed.")
    logger.info(f"df_train looks like: {df_train.head()}")

    return df_train, df_val, df_test
