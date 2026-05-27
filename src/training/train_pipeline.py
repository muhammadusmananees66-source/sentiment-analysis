"""
Distributed training pipeline with Kubeflow integration
"""

import mlflow
import optuna
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
import joblib
import pickle
import json
from datetime import datetime

class SentimentTrainer:
    """
    Production training pipeline with:
    - Multiple model architectures
    - Hyperparameter optimization (Optuna)
    - Cross-validation
    - MLflow tracking
    - Model registry integration
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.best_params = None

        # Set MLflow tracking
        mlflow.set_tracking_uri(config.get('mlflow_uri', 'http://localhost:5000'))
        mlflow.set_experiment(config.get('experiment_name', 'sentiment_analysis'))

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        run_hyperopt: bool = True
    ) -> Dict[str, Any]:
        """
        Train sentiment model with MLflow tracking

        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            run_hyperopt: Whether to run hyperparameter optimization
        """

        with mlflow.start_run(run_name=f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):

            # Log training config
            mlflow.log_params(self.config)
            mlflow.log_params({
                'train_samples': X_train.shape[0],
                'train_features': X_train.shape[1],
                'val_samples': X_val.shape[0]
            })

            # Hyperparameter optimization
            if run_hyperopt:
                self.best_params = self._optimize_hyperparameters(X_train, y_train)
                mlflow.log_params(self.best_params)
            else:
                self.best_params = self.config.get('default_params', {})

            # Train final model
            model_type = self.config.get('model_type', 'xgboost')
            self.model = self._train_model(X_train, y_train, model_type, self.best_params)

            # Evaluate
            metrics = self._evaluate(self.model, X_val, y_val)
            mlflow.log_metrics(metrics)

            # Log feature importance if available
            self._log_feature_importance(self.model, X_train)

            # Save model with MLflow
            mlflow.sklearn.log_model(self.model, "model")

            # Register model if validation accuracy meets threshold
            if metrics['accuracy'] >= self.config.get('min_accuracy', 0.75):
                model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
                mlflow.register_model(model_uri, "sentiment_classifier")

            return {
                'model': self.model,
                'metrics': metrics,
                'best_params': self.best_params
            }

    def _optimize_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict:
        """Bayesian hyperparameter optimization with Optuna"""

        model_type = self.config.get('model_type', 'xgboost')

        def objective(trial):
            if model_type == 'xgboost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 500, step=50),
                    'max_depth': trial.suggest_int('max_depth', 3, 12),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 10)
                }
                model = xgb.XGBClassifier(**params, random_state=42, use_label_encoder=False,
                                          eval_metric='logloss')

            elif model_type == 'lightgbm':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 500, step=50),
                    'max_depth': trial.suggest_int('max_depth', -1, 15),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'num_leaves': trial.suggest_int('num_leaves', 31, 255),
                    'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
                }
                model = lgb.LGBMClassifier(**params, random_state=42)

            elif model_type == 'logistic_regression':
                params = {
                    'C': trial.suggest_float('C', 0.01, 10, log=True),
                    'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
                    'solver': 'saga' if params.get('penalty') == 'l1' else 'lbfgs'
                }
                model = LogisticRegression(**params, random_state=42, max_iter=1000)

            else:  # random_forest
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 500, step=50),
                    'max_depth': trial.suggest_int('max_depth', 5, 30),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10)
                }
                model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)

            # Cross-validation score
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')

            return scores.mean()

        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=self.config.get('n_trials', 30), show_progress_bar=True)

        return study.best_params

    def _train_model(self, X_train, y_train, model_type, params):
        """Train final model with best parameters"""
        if model_type == 'xgboost':
            model = xgb.XGBClassifier(**params, random_state=42, use_label_encoder=False,
                                      eval_metric='logloss', n_jobs=-1)
        elif model_type == 'lightgbm':
            model = lgb.LGBMClassifier(**params, random_state=42, n_jobs=-1)
        elif model_type == 'logistic_regression':
            model = LogisticRegression(**params, random_state=42, max_iter=1000, n_jobs=-1)
        else:
            model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)

        model.fit(X_train, y_train)
        return model

    def _evaluate(self, model, X_val, y_val) -> Dict[str, float]:
        """Evaluate model performance"""
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val) if hasattr(model, 'predict_proba') else None

        precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_pred, average='weighted')

        metrics = {
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

        if y_pred_proba is not None and len(np.unique(y_val)) == 2:
            metrics['roc_auc'] = roc_auc_score(y_val, y_pred_proba[:, 1])

        return metrics

    def _log_feature_importance(self, model, X_train):
        """Log feature importance to MLflow"""
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            # Log top 20 features
            top_indices = np.argsort(importances)[-20:]
            for idx in top_indices:
                mlflow.log_metric(f"feature_imp_{idx}", importances[idx])
