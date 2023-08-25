import os
import sys
import warnings
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from deepchecks.core.suite import SuiteResult
from deepchecks.tabular import Dataset
from deepchecks.tabular.suites import model_evaluation, train_test_validation

from src.common.constants import (
    ARTIFACT_PATH,
    DATA_PATH,
    DRIFT_DETECTION_PATH,
    LOG_FILEPATH,
)
from src.common.logger import handle_exception, set_logger
from src.preprocess import CAT_FEATURES, preprocess_pipeline

logger = set_logger(os.path.join(LOG_FILEPATH, "logs.log"))
sys.excepthook = handle_exception
warnings.filterwarnings(action="ignore")


DATE = datetime.now().strftime("%Y%m%d")
LABEL_NAME = "rent"
model = joblib.load(os.path.join(ARTIFACT_PATH, "model.pkl"))


def load_data(filename: str) -> pd.DataFrame:
    return pd.read_csv(
        os.path.join(DATA_PATH, filename),
        usecols=lambda x: x not in ["area_locality", "posted_on", "id"],
    )


def log_failed_check_info(suite_result: SuiteResult):
    for result in suite_result.get_not_passed_checks():
        logger.info(
            "The following test failed!\n"
            f"{result.header}: {result.conditions_results[0].details}"
        )


def data_drift_detection(
    train_df: pd.DataFrame, new_df: pd.DataFrame, label: str, cat_features: str
) -> None:
    # Dataset 클래스를 이용해 train_set과 new_set을 만들 것
    train_set = Dataset(train_df, label=label, cat_features=cat_features)
    new_set = Dataset(new_df, label=label, cat_features=cat_features)

    validation_suite = train_test_validation()
    # Data Drift 결과를 얻기 위해 suite 실행
    suite_result = validation_suite.run(train_set, new_set)

    log_failed_check_info(suite_result=suite_result)

    suite_result.save_as_html(
        os.path.join(DRIFT_DETECTION_PATH, f"{DATE}_data_drift.html")
    )


def model_drift_detection(
    train_df: pd.DataFrame, new_df: pd.DataFrame, label: str, cat_features: str
) -> None:
    def get_xy(df: pd.DataFrame):
        y = np.log1p(df[LABEL_NAME])
        x = preprocess_pipeline.fit_transform(
            X=df.drop([LABEL_NAME], axis=1), y=y
        )

        return x, y

    x_train, y_train = get_xy(train_df)
    x_new, y_new = get_xy(new_df)

    train_set = Dataset(
        x_train,
        label=y_train,
        cat_features=CAT_FEATURES,
    )
    new_set = Dataset(
        x_new,
        label=y_new,
        cat_features=CAT_FEATURES,
    )

    evaluation_suite = model_evaluation()

    # Model Drift 결과를 얻기 위해 suite 실행
    suite_result = evaluation_suite.run(train_set, new_set, model["Regressor"])

    log_failed_check_info(suite_result=suite_result)

    suite_result.save_as_html(
        os.path.join(DRIFT_DETECTION_PATH, f"{DATE}_model_drift.html")
    )


def main():
    train_df = load_data(filename="house_rent_train.csv")
    new_df = load_data(filename="house_rent_new.csv")

    logger.debug(f"{train_df.info()}")
    logger.debug(f"{new_df.info()}")

    logger.info("Detect data drift")
    data_drift_detection(
        # Data drift detection 함수 인자 추가
        train_df=train_df,
        new_df=new_df,
        label=LABEL_NAME,
        cat_features=CAT_FEATURES,
    )

    logger.info("Detect model drift")
    model_drift_detection(
        # Model drift detection 함수 인자 추가
        train_df=train_df,
        new_df=new_df,
        label=LABEL_NAME,
        cat_features=CAT_FEATURES,
    )

    logger.info(
        "Detection results can be found in the following path:\n"
        f"{DRIFT_DETECTION_PATH}"
    )


if __name__ == "__main__":
    main()
