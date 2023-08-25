import logging
import logging.handlers
import os
import sys

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import seaborn as sns
from rich.logging import RichHandler
from sklearn.ensemble import GradientBoostingRegressor

from .constants import ARTIFACT_PATH, LOG_FILEPATH

RICH_FORMAT = "| %(filename)s:%(lineno)s\t| %(message)s"
FILE_HANDLER_FORMAT = (
    "[%(asctime)s]\t%(levelname)s\t | %(filename)s:%(lineno)s\t| %(message)s"
)


# file handler를 설정하고 가져오는 함수
def get_file_handler(
    log_path: str = LOG_FILEPATH,
) -> logging.handlers.TimedRotatingFileHandler:
    """로그 저장 파일 핸들러를 설정하는 함수

    Args:
        log_path (str, optional): 로그 저장 파일명. Defaults to LOG_FILEPATH.

    Returns:
        logging.handlers.TimedRotatingFileHandler: 로그 저장 파일 핸들러 객체
    """
    # TimedRotatingFileHandler 얘가 하는 역할 : log 파일이 있을 때 특정 기간을 두고 bakcup을 계속 해줌. 기간 이후 파일은 알아서 삭제하고...
    # interval : day / 1 : 매일 마다
    # backupcount : 30개 저장
    file_handler = logging.handlers.TimedRotatingFileHandler(
        log_path, when="midnight", interval=1, backupCount=30, encoding="utf-8"
    )
    file_handler.suffix = "logs-%Y%m%d"
    # 파일 핸들러의 기본 수준을 INFO로 설정
    file_handler.setLevel(logging.INFO)

    # 파일 핸들러의 포맷을 FILE_HANDLER_FORMAT으로 설정
    file_handler.setFormatter(logging.Formatter(FILE_HANDLER_FORMAT))

    return file_handler


def set_logger(log_path: str = LOG_FILEPATH) -> logging.Logger:
    """로거 설정 함수

    Args:
        log_path (str, optional): 로그 저장 파일명. Defaults to LOG_FILEPATH.

    Returns:
        logging.Logger: 설정한 로거 객체
    """
    logging.basicConfig(
        level="NOTSET",
        format=RICH_FORMAT,
        handlers=[RichHandler(rich_tracebacks=True)],
    )

    logger = logging.getLogger("rich")

    # 로거의 기본 수준을 DEBUG 설정
    logger.setLevel(logging.DEBUG)

    # TODO: 기본 로거에 위에서 만든 파일 핸들러를 추가
    logger.addHandler(get_file_handler(log_path))

    return logger


# 이 함수를 이용하여 try except 사용하지 않고 코드 관리
def handle_exception(exc_type, exc_value, exc_traceback):
    """Exception을 처리하는 함수입니다.
    이미 선언된 `logger`가 있을 때, 해당 `logger` 정보를 가져와서
    발생하는 Exception을 처리합니다.
    이 때, `KeyboardInterrupt`로 발생하는 예외는 처리하지 않습니다.
    """
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    if logging.root.manager.loggerDict:
        logger = logging.getLogger("rich")
    else:
        logger = set_logger(LOG_FILEPATH)

    logger.error(
        "Unexpected exception", exc_info=(exc_type, exc_value, exc_traceback)
    )
    logger.error("Unexpected exception caught!")


def log_feature_importance(
    train: pd.DataFrame, model: GradientBoostingRegressor
) -> None:
    """Scikit-learn 기반 모델의 피처 중요도를 차트로 저장합니다.

    Args:
        train (pd.DataFrame): 학습 데이터
        model (GradientBoostingRegressor): 모델
    """
    feature_imp = (
        pd.DataFrame(
            model.feature_importances_,
            index=train.columns,
            columns=["Importance"],
        )
        .sort_values(by="Importance", ascending=False)
        .reset_index(drop=False)
    )

    plt.figure(figsize=(12, 8))
    sns.barplot(
        data=feature_imp,
        x="Importance",
        y="index",
    ).set_title("Feature Importance")

    plt.savefig(
        os.path.join(ARTIFACT_PATH, "feature_importance.png"),
        bbox_inches="tight",
    )
    plt.close()
