import numpy as np
import pandas as pd
from category_encoders import TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer

CAT_FEATURES = [
    "area_type",
    "city",
    "furnishing_status",
    "tenant_preferred",
    "point_of_contact",
]


def extract_floor(floor_info: str) -> int:
    """층수 컬럼에서 실제 층수만 추출합니다.

    현재 층수 정보는 'X out of Y'로 되어 있습니다.
    여기에서 X만 추출하여 정수로 반환합니다.
    Upper basement, Lower basement, Ground out 등은 모두 0층으로 변환합니다.

    Args:
        floor_info (str): 층수 정보
    """
    split_floor_info = floor_info.split()
    floor_str = split_floor_info[0]

    return int(floor_str) if floor_str.isnumeric() else 0


def floor_extractor(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """`extract_floor()` 함수를 `FunctionTransformer`에 사용하기 위한
    Wrapping function입니다.

    Args:
        df (pd.DataFrame): 데이터프레임
        col (str): `extract_floor()`를 적용할 컬럼명
            `Floor`만 사용해야 함

    Returns:
        pd.DataFrame: 컬럼 처리 후 데이터
    """
    df[col] = df[col].apply(lambda x: extract_floor(x))
    return df


# 1. 방의 크기는 제곱근을 적용함 (FunctionTransformer 사용)
# 2. 층수는 실제 층수를 추출하되 숫자가 아닌 Basement 등은 0층으로 표기함
# 3. 범주형 변수(CAT_FEATURES)는 타겟 인코딩 적용 (from category_encoders import TargetEncoder)
preprocess_pipeline = ColumnTransformer(
    transformers=[
        ("sqrt_transformer", FunctionTransformer(np.sqrt), ["size"]),
        (
            "floor_extractor",
            FunctionTransformer(floor_extractor, kw_args={"col": "floor"}),
            ["floor"],
        ),
        ("target_encoder", TargetEncoder(), CAT_FEATURES),
    ],
    remainder="passthrough",
    verbose_feature_names_out=False,
)
preprocess_pipeline.set_output(transform="pandas")
