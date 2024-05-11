import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.metrics import make_scorer


def _get_preprocessor(numeric_model_features, categorical_model_features, standard_scale=False):

    if standard_scale == True:
        numeric_transformer = Pipeline(
            steps=[("imputer", SimpleImputer(add_indicator=True)), ("scale", StandardScaler())]
        )

    else:
        numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(add_indicator=True)),])

    categorical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="constant", fill_value="missing", add_indicator=False),
            ),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_model_features),
            ("categorical", categorical_transformer, categorical_model_features),
        ]
    )
    return preprocessor


def get_model_search_clf(
    model_type,
    numeric_model_features,
    categorical_model_features,
    n_groups=3,
    scoring_func="neg_log_loss",
    n_jobs=1,
):

    """
    Returns: 
        - GridSearch or RandomSearch estimator that implements fit() and get_best_estimator_. Also has GroupKFold cv strategy
    """

    scoring_func = "neg_log_loss"
    if model_type == "GBM":

        preprocessor = _get_preprocessor(
            numeric_model_features, categorical_model_features, standard_scale=False
        )
        pipe = Pipeline(
            [
                ("preproc", preprocessor),
                ("clf", GradientBoostingClassifier(n_estimators=1000, n_iter_no_change=25)),
            ]
        )
        params = {"clf__max_depth": [1, 2, 3], "clf__learning_rate": [0.1, 0.01]}

    if model_type == "HistGBM__test":

        preprocessor = _get_preprocessor(
            numeric_model_features, categorical_model_features, standard_scale=False
        )
        pipe = Pipeline(
            [("preproc", preprocessor), ("clf", HistGradientBoostingClassifier(max_iter=5))]
        )
        params = {"clf__max_depth": [1, 3, 5], "clf__learning_rate": [0.5, 0.1, 0.01]}

    if model_type == "HistGBM":

        preprocessor = _get_preprocessor(
            numeric_model_features, categorical_model_features, standard_scale=False
        )
        pipe = Pipeline(
            [("preproc", preprocessor), ("clf", HistGradientBoostingClassifier(max_iter=500))]
        )
        params = {"clf__max_depth": [1, 3, 5], "clf__learning_rate": [0.5, 0.1, 0.01]}

    if model_type == "HistGBMmonotone":

        """
            Enforcing that all features have a positive relationship with outcome because why not
        """
        preprocessor = _get_preprocessor(
            numeric_model_features, categorical_model_features, standard_scale=False
        )

        montone_constraints = [1 for i in np.arange(len(numeric_model_features))]
        pipe = Pipeline(
            [
                ("preproc", preprocessor),
                (
                    "clf",
                    HistGradientBoostingClassifier(max_iter=500, monotonic_cst=montone_constraints),
                ),
            ]
        )
        params = {"clf__max_depth": [1, 3, 5], "clf__learning_rate": [0.5, 0.1, 0.01]}

    if model_type == "RandomForest":

        """
            Enforcing that all features have a positive relationship with outcome because why not
        """
        preprocessor = _get_preprocessor(
            numeric_model_features, categorical_model_features, standard_scale=False
        )

        montone_constraints = [1 for i in np.arange(len(numeric_model_features))]
        pipe = Pipeline(
            [
                ("preproc", preprocessor),
                ("clf", RandomForestClassifier(n_estimators=100, n_jobs=n_jobs)),
            ]
        )
        params = {"clf__max_features": [0.1, 0.5], "clf__max_depth": [3, 5, None]}

    if model_type == "HistGBM_randomCV":

        from scipy.stats import lognorm, randint

        preprocessor = _get_preprocessor(
            numeric_model_features, categorical_model_features, standard_scale=False
        )
        pipe = Pipeline(
            [("preproc", preprocessor), ("clf", HistGradientBoostingClassifier(max_iter=500))]
        )

        params = {
            "clf__max_depth": [None, 2, 3, 4],
            "clf__learning_rate": lognorm(s=0.5, scale=np.exp(-4), loc=np.exp(-7)),
            "clf__max_leaf_nodes": randint(5, 31),
            "clf__min_samples_leaf": randint(2, 20),
        }

    if model_type == "HistGBM_precision_opt":

        preprocessor = _get_preprocessor(
            numeric_model_features, categorical_model_features, standard_scale=False
        )
        pipe = Pipeline(
            [
                ("preproc", preprocessor),
                (
                    "clf",
                    HistGradientBoostingClassifier(
                        max_iter=500,
                        scoring=make_scorer(get_precision_at_thresholds, greater_is_better=True),
                    ),
                ),
            ]
        )
        params = {"clf__max_depth": [1, 3, 5], "clf__learning_rate": [0.5, 0.1, 0.01]}

    if model_type == "logistic":
        preprocessor = _get_preprocessor(
            numeric_model_features, categorical_model_features, standard_scale=True
        )
        pipe = Pipeline(
            [
                ("preproc", preprocessor),
                ("clf", LogisticRegression(penalty="elasticnet", solver="saga", max_iter=500)),
            ]
        )
        params = {
            "clf__l1_ratio": [0.1, 0.5, 0.9],
            "clf__C": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
        }
    if model_type == "nonnegative_LPM":
        preprocessor = _get_preprocessor(
            numeric_model_features, categorical_model_features, standard_scale=True
        )
        pipe = Pipeline([("preproc", preprocessor), ("clf", ElasticNet(positive=True))])
        params = {"clf__l1_ratio": [0.1, 0.5, 0.9], "clf__alpha": [0.01, 1, 10, 500, 1000, 10000]}
        scoring_func = "roc_auc"
    if model_type == "dummy":
        from sklearn.dummy import DummyClassifier

        preprocessor = _get_preprocessor(
            numeric_model_features, categorical_model_features, standard_scale=False
        )
        pipe = Pipeline([("preproc", preprocessor), ("clf", DummyClassifier())])
        params = {
            "clf__strategy": ["prior", "uniform"],
        }

    gkf = StratifiedGroupKFold(n_splits=n_groups)

    if model_type == "HistGBM_precision_opt":
        model = GridSearchCV(
            pipe,
            params,
            cv=gkf,
            n_jobs=n_jobs,
            scoring=make_scorer(get_precision_at_thresholds, greater_is_better=True),
        )
    elif model_type == "HistGBM_randomCV":
        model = RandomizedSearchCV(
            pipe, params, cv=gkf, n_jobs=n_jobs, scoring=scoring_func, n_iter=50
        )

    else:

        model = GridSearchCV(pipe, params, cv=gkf, n_jobs=n_jobs, scoring=scoring_func)

    return model


def _get_pseudo_id(data, random_state):

    import random

    random.seed(random_state)
    emp_id_list = data["tax_id"].unique().tolist()
    random.shuffle(emp_id_list)

    mapper = {emp_id: idx for idx, emp_id in enumerate(emp_id_list)}
    return data["tax_id"].map(mapper)


def get_precision_at_x_pct(y_flag, y_pred, threshold):
    """
    """
    precision_rank = int(threshold * len(y_flag))
    prediction_order = np.argsort(y_pred)[::-1][:precision_rank]
    return sum(np.asarray(y_flag)[prediction_order]) / precision_rank


def get_precision_at_thresholds(y_true, y_pred, thresholds=[0.01, 0.05, 0.02, 0.1]):
    """
    """
    y_flag = (y_true >= 1) * 1.0
    return sum([get_precision_at_x_pct(y_flag, y_pred, thresh) for thresh in thresholds]) / len(
        thresholds
    )
