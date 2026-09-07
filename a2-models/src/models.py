"""Model + preprocessing pipelines shared by every A2 script.

Preprocessing mirrors A1's ColumnTransformer exactly (median impute + Yeo-Johnson +
RobustScaler for numeric, constant-UNK impute + one-hot for categorical). Everything
lives inside one imblearn ``Pipeline`` so that both preprocessing statistics and any
resampling step are refit fresh on the training fold only.
"""
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, RobustScaler
from xgboost import XGBClassifier

from src.data import CATEGORY_COLS, NUMERICAL_COLS

RANDOM_STATE = 42


def build_preprocessor(num_cols=NUMERICAL_COLS, cat_cols=CATEGORY_COLS):
    numeric_steps = ImbPipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("power", PowerTransformer(method="yeo-johnson")),
        ("scale", RobustScaler()),
    ])
    category_steps = ImbPipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="UNK")),
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=50)),
    ])
    return ColumnTransformer([
        ("num", numeric_steps, num_cols),
        ("cat", category_steps, cat_cols),
    ])


def make_classifier(name: str, params: dict | None = None, imbalance: str = "class_weight"):
    params = dict(params or {})
    if name == "logreg":
        cw = "balanced" if imbalance == "class_weight" else None
        return LogisticRegression(max_iter=2000, class_weight=cw, random_state=RANDOM_STATE, **params)
    if name == "random_forest":
        cw = "balanced" if imbalance == "class_weight" else None
        return RandomForestClassifier(class_weight=cw, random_state=RANDOM_STATE, n_jobs=-1, **params)
    if name == "xgboost":
        return XGBClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1,
            eval_metric="aucpr",
            tree_method="hist",
            **params,
        )
    raise ValueError(name)


def build_pipeline(name: str, params: dict | None = None, imbalance: str = "class_weight",
                    scale_pos_weight: float | None = None):
    """imbalance in {'class_weight', 'smote', 'none'}."""
    params = dict(params or {})
    if name == "xgboost" and imbalance == "class_weight" and scale_pos_weight is not None:
        params["scale_pos_weight"] = scale_pos_weight

    clf = make_classifier(name, params, imbalance)
    steps = [("prep", build_preprocessor())]
    if imbalance == "smote":
        steps.append(("resample", SMOTE(random_state=RANDOM_STATE)))
    steps.append(("clf", clf))
    return ImbPipeline(steps)
