#!/usr/bin/env python3

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--sqlite-database", required=True,
                    help="Filename to read the experiments from")
parser.add_argument("--force-retrain", action="store_true",
                    help="Force retraining, even if it is unnecessary")
args = parser.parse_args()

import sys
import sqlite3
import pandas
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import PolynomialFeatures
import sklearn.model_selection
import joblib
import numpy
from io import BytesIO
import differencer
import typescorers

conn = sqlite3.connect(args.sqlite_database)
cursor = conn.cursor()
cursor.execute("pragma foreign_keys = on;")
cursor.execute("""create table if not exists bmap3_training (
  type_1_error_ratio float,
  type_2_error_ratio float,
  roc_auc float,
  accuracy float,
  last_trained timestamp);
""")
cursor.execute("select control_group_size, experiment_group_size, most_recent_write from metadata")
row = cursor.fetchone()
if row is None:
    sys.exit(f"Metadata in {args.sqlite_database} has not been initialized correctly")
control_group_size, experiment_group_size, most_recent_write = row

if not(args.force_retrain):
    cursor.execute("select last_trained from bmap3_training")
    row = cursor.fetchone()
    if row is not None:
        last_trained = row[0]
        if most_recent_write < last_trained:
            sys.exit("Nothing to do")


control_columns = [f"control{i+1}" for i in range(control_group_size)]
experiment_columns = [f"experimental{i+1}" for i in range(experiment_group_size)]
select_args = ", ".join(control_columns + experiment_columns)

raw_data_dataframe = pandas.read_sql(f"select {select_args}, was_successful from synthetic_experiments", conn)

custom_transformer = differencer.ControlExperimentalDifferencer(control_group_size, experiment_group_size)
poly_features = PolynomialFeatures(degree=2)
log_reg = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000, class_weight='balanced')
#calibrated_clf = CalibratedClassifierCV(base_estimator=log_reg, method='sigmoid')

# Constructing the pipeline
pipeline = Pipeline([
    ('feature_difference', custom_transformer),
    ('poly_features', poly_features),
    #('classifier', calibrated_clf)
    ('classifier', log_reg)
])


scoring = sklearn.model_selection.cross_validate(
    pipeline,
    raw_data_dataframe[control_columns + experiment_columns],
    raw_data_dataframe.was_successful,
    scoring={
        'type_1_error_ratio': typescorers.type_1_error_scorer,
        'type_2_error_ratio': typescorers.type_2_error_scorer,
        'accuracy': sklearn.metrics.make_scorer(sklearn.metrics.accuracy_score),
        'roc_auc': sklearn.metrics.make_scorer(sklearn.metrics.roc_auc_score)
    }
)

mean_scores = {metric: numpy.mean(values) for metric, values in scoring.items()}
data_to_insert = (
    mean_scores['test_type_1_error_ratio'],
    mean_scores['test_type_2_error_ratio'],
    mean_scores['test_roc_auc'],
    mean_scores['test_accuracy'],
)

cursor.execute("delete from bmap3_training;")
cursor.execute("""INSERT OR REPLACE INTO bmap3_training
    (type_1_error_ratio, type_2_error_ratio, roc_auc, accuracy, last_trained)
    VALUES (?, ?, ?, ?, current_timestamp);
""", data_to_insert)
conn.commit()
                                       
pipeline.fit(raw_data_dataframe[control_columns + experiment_columns],
             raw_data_dataframe.was_successful)

cursor.execute("""
CREATE TABLE IF NOT EXISTS bmap3_storage (
    model BLOB,
    saved_on TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);""")
cursor.execute("delete from bmap3_storage")


# Serialize the model
model_buffer = BytesIO()
joblib.dump(pipeline, model_buffer)
model_buffer.seek(0)  # Rewind the buffer to the beginning of the stream
serialized_model = model_buffer.getvalue()

# Insert the model into the database
insert_query = """
    INSERT model_storage (model)
    VALUES (?, ?);
"""
cursor.execute("insert into bmap3_storage (model) values (?)", (serialized_model,))
conn.commit()
