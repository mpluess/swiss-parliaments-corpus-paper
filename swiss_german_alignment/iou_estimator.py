from lightgbm import LGBMRegressor
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from skopt import BayesSearchCV
from skopt.space.space import Integer


class IouEstimator:
    FEATURE_PREFIX = 'feature__'
    LABEL_PREFIX = 'label__'
    PREDICTION_PREFIX = 'prediction__'

    DEFAULT_HYPERPARAMS = {
        'num_leaves': 3,
        'min_child_samples': 7,
        'max_bin': 7597,
    }

    def __init__(self, optimize_hyperparams, hyperparams: dict = None):
        self.optimize_hyperparams = optimize_hyperparams
        self.hyperparams = hyperparams

        self.estimator_ = None

    def fit(self, df: pd.DataFrame):
        df_features = self._to_feature_df(df, True)

        df_features = df_features.dropna()
        df_features = df_features.sample(frac=1, random_state=42)

        X = self._get_X(df_features)
        y = self._get_y(df_features)

        if self.optimize_hyperparams:
            def scorer(estimator, X, y):
                y_pred = np.clip(np.squeeze(estimator.predict(X)), 0.0, 1.0)
                return -mean_absolute_error(y, y_pred)

            print(f'IouEstimator: optimizing hyperparams with Bayesian Optimization')
            opt = BayesSearchCV(
                LGBMRegressor(),
                {
                    'num_leaves': Integer(
                        2, 128,
                        prior='log-uniform', base=2
                    ),
                    'min_child_samples': Integer(
                        2, 512,
                        prior='log-uniform', base=2
                    ),
                    'max_bin': Integer(
                        2, 8192,
                        prior='log-uniform', base=2
                    ),
                },
                n_iter=60,
                optimizer_kwargs={
                    'n_initial_points': 20,
                    'base_estimator': 'GP',
                },
                scoring=scorer,
                cv=3,
                refit=False,
                random_state=42,
                return_train_score=True,
            )
            opt.fit(X, y)
            print(f'Found hyperparams {opt.best_params_}')
            print(f"Train score: {opt.cv_results_['mean_train_score'][opt.best_index_]}")
            print(f'Test score: {opt.best_score_}')
            estimator = LGBMRegressor(**opt.best_params_)
        elif self.hyperparams is not None:
            print(f'IOUEstimator: using using hyperparams {self.hyperparams}')
            estimator = LGBMRegressor(**self.hyperparams)
        else:
            print(f'IOUEstimator: using default hyperparams {self.DEFAULT_HYPERPARAMS}')
            estimator = LGBMRegressor(**self.DEFAULT_HYPERPARAMS)

        self.estimator_ = estimator.fit(X, y)

        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        df_features = self._to_feature_df(df, False)

        df_dropna = df_features.dropna().copy()
        key = f'{self.PREDICTION_PREFIX}iou_estimate'
        df_dropna[key] = np.clip(np.squeeze(self.estimator_.predict(self._get_X(df_dropna))), 0.0, 1.0)

        # Undo dropna
        df_features = df_features.join(df_dropna, how='left', lsuffix='l')

        return df_features[key]

    def _to_feature_df(self, df, with_label):
        df = df.copy()

        df[f'{self.FEATURE_PREFIX}length_ratio'] = df.apply(lambda row: row['truth_length'] / row['stt_length'] if row['stt_length'] > 0 else 100, axis=1)
        df[f'{self.FEATURE_PREFIX}score_normalized'] = df.apply(lambda row: row['score'] / row['truth_length'] if row['score'] is not None and row['truth_length'] > 0 else None, axis=1)
        df = df.rename(columns={
            'stt_confidence': f'{self.FEATURE_PREFIX}stt_confidence',
        })
        df[f'{self.FEATURE_PREFIX}chars_per_second_truth'] = df.apply(lambda row: row['truth_length'] / row['audio_duration'] if row['audio_duration'] > 0 else 1000, axis=1)
        if with_label:
            df = df.rename(columns={
                'iou': f'{self.LABEL_PREFIX}iou',
            })

        attributes = [col for col in df.columns if col.startswith(self.FEATURE_PREFIX) or col.startswith(self.LABEL_PREFIX)]
        df = df[attributes].copy()

        return df

    def _get_X(self, df_features):
        return df_features[[col for col in df_features.columns if col.startswith(self.FEATURE_PREFIX)]].values

    def _get_y(self, df_features):
        return df_features[[col for col in df_features.columns if col.startswith(self.LABEL_PREFIX)][0]].values
