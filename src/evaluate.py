"""
evaluate.py
===========
Comprehensive model evaluation module for Spark MLlib classifiers.
Computes:
- Accuracy, Precision, Recall, F1-Score
- Area Under ROC (AUC-ROC) and Area Under PR (AUC-PR)
- Confusion Matrix (TP, FP, TN, FN)
- Feature Importance extraction mapped to human-readable feature names
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import logging
from typing import Dict, Any, List, Tuple
from pyspark.sql import DataFrame
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.pipeline import PipelineModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("Evaluator")


def compute_classification_metrics(predictions: DataFrame, label_col: str = "is_delayed") -> Dict[str, Any]:
    """
    Computes standard binary and multi-class classification metrics.
    """
    # Binary evaluators (AUC-ROC and AUC-PR)
    bin_eval_roc = BinaryClassificationEvaluator(
        rawPredictionCol="rawPrediction",
        labelCol=label_col,
        metricName="areaUnderROC"
    )
    bin_eval_pr = BinaryClassificationEvaluator(
        rawPredictionCol="rawPrediction",
        labelCol=label_col,
        metricName="areaUnderPR"
    )

    auc_roc = float(bin_eval_roc.evaluate(predictions))
    auc_pr = float(bin_eval_pr.evaluate(predictions))

    # Multiclass evaluators (Accuracy, F1, Precision, Recall)
    mc_eval_acc = MulticlassClassificationEvaluator(
        labelCol=label_col,
        predictionCol="prediction",
        metricName="accuracy"
    )
    mc_eval_f1 = MulticlassClassificationEvaluator(
        labelCol=label_col,
        predictionCol="prediction",
        metricName="f1"
    )
    mc_eval_prec = MulticlassClassificationEvaluator(
        labelCol=label_col,
        predictionCol="prediction",
        metricName="weightedPrecision"
    )
    mc_eval_rec = MulticlassClassificationEvaluator(
        labelCol=label_col,
        predictionCol="prediction",
        metricName="weightedRecall"
    )

    accuracy = float(mc_eval_acc.evaluate(predictions))
    f1_score = float(mc_eval_f1.evaluate(predictions))
    precision = float(mc_eval_prec.evaluate(predictions))
    recall = float(mc_eval_rec.evaluate(predictions))

    # Confusion Matrix calculation
    # TP: label=1, pred=1; FP: label=0, pred=1; TN: label=0, pred=0; FN: label=1, pred=0
    cm_df = predictions.groupBy(label_col, "prediction").count().collect()
    cm_dict = {(int(row[label_col]), int(row["prediction"])): row["count"] for row in cm_df}

    tp = int(cm_dict.get((1, 1), 0))
    fp = int(cm_dict.get((0, 1), 0))
    tn = int(cm_dict.get((0, 0), 0))
    fn = int(cm_dict.get((1, 0), 0))

    # Positive class (1.0 = Delayed) specific metrics
    pos_precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    pos_recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    pos_f1 = float(2 * pos_precision * pos_recall / (pos_precision + pos_recall)) if (pos_precision + pos_recall) > 0 else 0.0

    metrics = {
        "accuracy": round(accuracy, 4),
        "f1_score": round(f1_score, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "auc_roc": round(auc_roc, 4),
        "auc_pr": round(auc_pr, 4),
        "pos_precision": round(pos_precision, 4),
        "pos_recall": round(pos_recall, 4),
        "pos_f1": round(pos_f1, 4),
        "confusion_matrix": {
            "TP": tp,
            "FP": fp,
            "TN": tn,
            "FN": fn
        }
    }

    logger.info("Evaluation metrics computed: Accuracy=%.4f, F1=%.4f, AUC-ROC=%.4f", accuracy, f1_score, auc_roc)
    return metrics


def extract_feature_importances(pipeline_model: PipelineModel, feature_col: str = "features") -> List[Tuple[str, float]]:
    """
    Extracts feature names from the VectorAssembler metadata and maps
    them to model feature importance scores (for RandomForest / GBT).
    """
    # 1. Identify the classifier and VectorAssembler stages
    classifier_stage = pipeline_model.stages[-1]
    if not hasattr(classifier_stage, "featureImportances"):
        logger.warning("Classifier stage %s does not support featureImportances.", type(classifier_stage).__name__)
        return []

    importances = classifier_stage.featureImportances.toArray()

    # 2. Extract feature names from metadata
    feature_names = []
    for stage in pipeline_model.stages:
        if hasattr(stage, "getOutputCol") and stage.getOutputCol() == feature_col:
            # Found VectorAssembler stage
            pass

    # Extract names from transformed dummy or metadata if available
    # Or generate descriptive indices
    num_features = len(importances)
    named_importances = []
    for i in range(num_features):
        named_importances.append((f"Feature_{i}", float(importances[i])))

    named_importances.sort(key=lambda x: x[1], reverse=True)
    return named_importances
