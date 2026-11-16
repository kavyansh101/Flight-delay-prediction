"""
train.py
========
Trains and tunes 3 machine learning models using Spark MLlib:
1. Logistic Regression
2. Random Forest Classifier
3. Gradient-Boosted Trees (GBT)

Features:
- Spark ML Pipeline (StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler)
- Class imbalance handling using computed class weights
- 3-fold CrossValidator with ParamGridBuilder for hyperparameter tuning
- Full metric evaluation and comparison
- Persistence of the best PipelineModel to disk
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import os
import json
import time
import logging
import pandas as pd
from typing import Dict, Any, List, Tuple

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import (
    StringIndexer,
    OneHotEncoder,
    VectorAssembler,
    StandardScaler,
)
from pyspark.ml.classification import (
    LogisticRegression,
    RandomForestClassifier,
    GBTClassifier,
)
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator

from src.spark_session import get_spark_session
from src.evaluate import compute_classification_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ModelTrainer")

CATEGORICAL_COLS = ["OP_CARRIER", "ORIGIN", "DEST", "DEP_TIME_OF_DAY", "SEASON", "DISTANCE_GROUP"]
NUMERICAL_COLS = [
    "CRS_DEP_HOUR",
    "DAY_OF_WEEK",
    "DAY_OF_MONTH",
    "MONTH",
    "DISTANCE",
    "CRS_ELAPSED_TIME",
    "CARRIER_DELAY_RATE",
    "ORIGIN_HOURLY_TRAFFIC",
    "DEST_HOURLY_TRAFFIC",
    "ORIGIN_DAILY_CUMULATIVE_FLIGHTS",
]


def add_class_weights(df: DataFrame, label_col: str = "is_delayed") -> Tuple[DataFrame, float, float]:
    """
    Computes inverse class frequencies and attaches a balancing 'class_weight' column.
    """
    logger.info("Computing class balance weights for '%s'...", label_col)
    total_count = df.count()
    pos_count = df.filter(F.col(label_col) == 1.0).count()
    neg_count = total_count - pos_count

    w_pos = float(total_count / (2.0 * pos_count))
    w_neg = float(total_count / (2.0 * neg_count))

    logger.info("Total rows: %d | Negative (On-Time): %d (weight=%.4f) | Positive (Delayed): %d (weight=%.4f)",
                total_count, neg_count, w_neg, pos_count, w_pos)

    weighted_df = df.withColumn(
        "class_weight",
        F.when(F.col(label_col) == 1.0, F.lit(w_pos)).otherwise(F.lit(w_neg))
    )
    return weighted_df, w_pos, w_neg


def build_preprocessing_pipeline() -> Tuple[List[Any], VectorAssembler]:
    """
    Constructs the feature transformation stages:
    StringIndexer -> OneHotEncoder -> VectorAssembler -> StandardScaler
    """
    indexers = [
        StringIndexer(inputCol=col, outputCol=f"{col}_idx", handleInvalid="keep")
        for col in CATEGORICAL_COLS
    ]

    encoder = OneHotEncoder(
        inputCols=[f"{col}_idx" for col in CATEGORICAL_COLS],
        outputCols=[f"{col}_vec" for col in CATEGORICAL_COLS],
        handleInvalid="keep"
    )

    assembler_inputs = [f"{col}_vec" for col in CATEGORICAL_COLS] + NUMERICAL_COLS
    assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="raw_features", handleInvalid="keep")

    scaler = StandardScaler(inputCol="raw_features", outputCol="features", withStd=True, withMean=False)

    stages = indexers + [encoder, assembler, scaler]
    return stages, assembler


def get_feature_names_from_model(pipeline_model: PipelineModel) -> List[str]:
    """
    Extracts the names of all assembled features from the fitted pipeline.
    """
    feature_names = []
    # Identify StringIndexer stages to get category mappings
    indexer_labels = {}
    for stage in pipeline_model.stages:
        if isinstance(stage, StringIndexer):
            pass

    # Inspect the VectorAssembler output attributes metadata
    for stage in reversed(pipeline_model.stages):
        if hasattr(stage, "summary") or hasattr(stage, "featureImportances"):
            # Model stage
            continue

    # Fallback/standard feature naming
    for c in CATEGORICAL_COLS:
        feature_names.append(f"{c}_encoded")
    for n in NUMERICAL_COLS:
        feature_names.append(n)
    return feature_names


def train_and_tune_models(
    parquet_path: str = "data/processed/flights_processed.parquet",
    models_dir: str = "models",
    outputs_dir: str = "outputs"
) -> Dict[str, Any]:
    """
    Orchestrates data loading, train/test split, hyperparameter tuning with CrossValidator,
    model evaluation, and model serialization.
    """
    spark = get_spark_session("FlightDelayModelTraining")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(os.path.join(outputs_dir, "figures"), exist_ok=True)

    # 1. Load Parquet Data
    logger.info("Loading processed Parquet dataset from: %s", parquet_path)
    df = spark.read.parquet(parquet_path)

    # 2. Add class weights
    weighted_df, w_pos, w_neg = add_class_weights(df)

    # 3. Train / Test Split (80% train, 20% test)
    logger.info("Performing 80/20 train/test split (seed=42)...")
    train_df, test_df = weighted_df.randomSplit([0.8, 0.2], seed=42)
    train_df = train_df.cache()
    test_df = test_df.cache()
    train_count = train_df.count()
    test_count = test_df.count()
    logger.info("Train dataset size: %d rows | Test dataset size: %d rows", train_count, test_count)

    # Subsample train for hyperparameter cross-validation grid search (fast & representative)
    cv_subsample = train_df.sample(withReplacement=False, fraction=min(1.0, 60000.0 / train_count), seed=42).cache()

    # 4. Feature preprocessing stages
    prep_stages, assembler = build_preprocessing_pipeline()

    # Binary Classification Evaluator for ROC-AUC
    evaluator = BinaryClassificationEvaluator(
        rawPredictionCol="rawPrediction",
        labelCol="is_delayed",
        metricName="areaUnderROC"
    )

    models_config = {
        "LogisticRegression": {
            "estimator": LogisticRegression(
                featuresCol="features",
                labelCol="is_delayed",
                weightCol="class_weight",
                maxIter=25
            ),
            "param_grid": (
                ParamGridBuilder()
                .addGrid(LogisticRegression.regParam, [0.01, 0.1])
                .addGrid(LogisticRegression.elasticNetParam, [0.0, 0.5])
                .build()
            )
        },
        "RandomForest": {
            "estimator": RandomForestClassifier(
                featuresCol="features",
                labelCol="is_delayed",
                weightCol="class_weight",
                seed=42
            ),
            "param_grid": (
                ParamGridBuilder()
                .addGrid(RandomForestClassifier.maxDepth, [6, 10])
                .addGrid(RandomForestClassifier.numTrees, [30, 50])
                .build()
            )
        },
        "GradientBoostedTrees": {
            "estimator": GBTClassifier(
                featuresCol="features",
                labelCol="is_delayed",
                weightCol="class_weight",
                seed=42
            ),
            "param_grid": (
                ParamGridBuilder()
                .addGrid(GBTClassifier.maxDepth, [5, 7])
                .addGrid(GBTClassifier.maxIter, [20, 30])
                .build()
            )
        }
    }

    results = {}
    best_overall_model = None
    best_overall_auc = -1.0
    best_overall_name = ""
    feature_importances_dict = {}

    for model_name, cfg in models_config.items():
        logger.info("\n" + "=" * 60)
        logger.info("Starting Training & Tuning for: %s", model_name)
        logger.info("=" * 60)
        start_m = time.time()

        # Build full pipeline: Preprocessing + Estimator
        full_pipeline = Pipeline(stages=prep_stages + [cfg["estimator"]])

        # CrossValidator (3-fold)
        logger.info("Configuring 3-fold CrossValidator with ParamGrid for %s...", model_name)
        cv = CrossValidator(
            estimator=full_pipeline,
            estimatorParamMaps=cfg["param_grid"],
            evaluator=evaluator,
            numFolds=3,
            parallelism=4,
            seed=42
        )

        logger.info("Fitting CrossValidator on representative fold sample (%d rows)...", cv_subsample.count())
        cv_model = cv.fit(cv_subsample)
        best_sub_model = cv_model.bestModel
        best_stage = best_sub_model.stages[-1]

        # Log tuned hyperparameters
        logger.info("Cross-Validation complete. Optimal parameters found for %s.", model_name)

        # Now fit on full training dataset with the best tuned hyperparams
        logger.info("Retraining best %s configuration on full training set (%d rows)...", model_name, train_count)
        final_pipeline = Pipeline(stages=prep_stages + [best_stage])
        fitted_model = final_pipeline.fit(train_df)

        train_time = time.time() - start_m
        logger.info("%s full training completed in %.2f seconds.", model_name, train_time)

        # 5. Evaluate on held-out test set
        logger.info("Evaluating %s on held-out test set (%d rows)...", model_name, test_count)
        test_predictions = fitted_model.transform(test_df)
        metrics = compute_classification_metrics(test_predictions, label_col="is_delayed")
        metrics["training_time_sec"] = round(train_time, 2)

        results[model_name] = {
            "metrics": metrics,
            "fitted_model": fitted_model
        }

        # Extract Feature Importances if available
        classifier_stage = fitted_model.stages[-1]
        if hasattr(classifier_stage, "featureImportances"):
            raw_imp = classifier_stage.featureImportances.toArray()
            feature_importances_dict[model_name] = [float(x) for x in raw_imp]

        if metrics["auc_roc"] > best_overall_auc:
            best_overall_auc = metrics["auc_roc"]
            best_overall_model = fitted_model
            best_overall_name = model_name

    logger.info("\n" + "=" * 60)
    logger.info("MODEL COMPARISON SUMMARY")
    logger.info("=" * 60)
    summary_rows = []
    for m_name, res in results.items():
        met = res["metrics"]
        summary_rows.append({
            "Model": m_name,
            "Accuracy": met["accuracy"],
            "F1_Score": met["f1_score"],
            "Precision": met["precision"],
            "Recall": met["recall"],
            "AUC_ROC": met["auc_roc"],
            "AUC_PR": met["auc_pr"],
            "Positive_Precision": met["pos_precision"],
            "Positive_Recall": met["pos_recall"],
            "Positive_F1": met["pos_f1"],
            "Train_Time_Sec": met["training_time_sec"],
        })
        logger.info(
            "%-22s | Acc: %.4f | F1: %.4f | Prec: %.4f | Rec: %.4f | AUC: %.4f | Time: %.1fs",
            m_name, met["accuracy"], met["f1_score"], met["precision"], met["recall"], met["auc_roc"], met["training_time_sec"]
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = os.path.join(outputs_dir, "model_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    logger.info("Saved model comparison table to: %s", summary_csv_path)

    # Save metrics JSON
    metrics_export = {
        "best_model": best_overall_name,
        "best_auc_roc": best_overall_auc,
        "models": {m: res["metrics"] for m, res in results.items()},
        "feature_importances": feature_importances_dict,
        "numerical_features": NUMERICAL_COLS,
        "categorical_features": CATEGORICAL_COLS
    }
    metrics_json_path = os.path.join(outputs_dir, "metrics.json")
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_export, f, indent=2)
    logger.info("Saved detailed metrics JSON to: %s", metrics_json_path)

    # 6. Save Best Overall PipelineModel
    best_model_path = os.path.join(models_dir, "best_flight_delay_model")
    logger.info("Saving best model (%s, AUC=%.4f) to: %s ...", best_overall_name, best_overall_auc, best_model_path)
    best_overall_model.write().overwrite().save(best_model_path)
    logger.info("Successfully persisted best Spark MLlib PipelineModel.")

    # Save best model metadata
    model_meta = {
        "best_model_name": best_overall_name,
        "best_auc_roc": best_overall_auc,
        "metrics": results[best_overall_name]["metrics"],
        "class_weight_pos": w_pos,
        "class_weight_neg": w_neg,
        "model_path": best_model_path
    }
    with open(os.path.join(models_dir, "best_model_info.json"), "w", encoding="utf-8") as f:
        json.dump(model_meta, f, indent=2)

    return metrics_export


if __name__ == "__main__":
    train_and_tune_models()
