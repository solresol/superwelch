#!/usr/bin/env python3

# Synthesize some number of random synthetic experiments, and
# put them into a SQLite database
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--sqlite-database", required=True,
                    help="Filename to store the experiments")
parser.add_argument("--progress", action='store_true', help="Show a progress bar")
parser.add_argument("--control-group-size", type=int, default=3, help="Size of the control group")
parser.add_argument("--experiment-group-size", type=int, default=3, help="Size of the experiment group")
args = parser.parse_args()
import scipy
import sqlite3
import sys
import numpy
from sklearn.metrics import roc_auc_score, accuracy_score

conn = sqlite3.connect(args.sqlite_database)
cursor = conn.cursor()
cursor.execute("pragma foreign_keys = on;")
cursor.execute("""create table if not exists metadata (
  control_group_size int,
  experiment_group_size int,
  proportion_that_do_nothing float,
  control_mean_low float,
  control_mean_high float,
  control_stddev_low float,
  control_stddev_high float,
  experimental_mean_low float,
  experimental_mean_high float,
  experimental_stddev_low float,
  experimental_stddev_high float,
  most_recent_write timestamp default current_timestamp);""")

cursor.execute("""SELECT  control_group_size, experiment_group_size FROM metadata LIMIT 1;""")
row = cursor.fetchone()

if row:
    # Unpack row values
    (
        control_group_size, experiment_group_size
    ) = row

    # Compare with command-line arguments
    mismatches = []
    arg_vars = vars(args)
    for field, value in zip(cursor.description, row):
        field_name = field[0]
        if arg_vars[field_name] != value:
            mismatches.append(f"{field_name} is {value} in the database, and {arg_vars[field_name]} on the command-line")

    if mismatches:
        mismatch_errors = "\n - ".join(mismatches)
        sys.exit(f"Mismatched parameters:\n - {mismatch_errors}")
else:
    # Insert new row
    insert_query = """
    INSERT INTO metadata (
        control_group_size, experiment_group_size, proportion_that_do_nothing,
        control_mean_low, control_mean_high, control_stddev_low, control_stddev_high,
        experimental_mean_low, experimental_mean_high, experimental_stddev_low, experimental_stddev_high
    ) VALUES (?, ?, 0.5, 3.5, 3.5, 1.87, 1.87, 4.5, 4.5, 2.45, 2.45);"""
    
    cursor.execute(insert_query, (
        args.control_group_size, args.experiment_group_size
    ))

    conn.commit()


control_columns = [f"control{i+1}" for i in range(args.control_group_size)]
experiment_columns = [f"experimental{i+1}" for i in range(args.experiment_group_size)]
insert_args = ", ".join(control_columns + experiment_columns)
insert_params = ",".join(["?" for t in (control_columns + experiment_columns)])
create_defs = [f"{x} float" for x in (control_columns + experiment_columns)]
columns = "\n, ".join(create_defs)

cursor.execute(f"""create table if not exists synthetic_experiments (
 synthetic_experiment_id integer primary key,
 control_mean float,
 control_stddev float,
 experiment_mean float,
 experiment_stddev float,
 {columns},
 was_successful bool
);""")


assert(args.control_group_size == 3)
assert(args.experiment_group_size == 3)

import tqdm
for d1 in tqdm.tqdm(range(1,7)):
    for d2 in range(d1,7):
        for d3 in range(d2,7):
            for d4 in range(1,7):
                for d5 in range(d4,7):
                    for d6 in range(d5,7):
                        outcome = scipy.stats.ttest_ind([d1,d2,d3], [d4,d5,d6], equal_var=False)
                        cursor.execute(f"insert into synthetic_experiments (control_mean, control_stddev, experiment_mean, experiment_stddev, {insert_args}, was_successful) values (3.5, 1.87, 3.5, 1.87, ?,?,?,?,?,?, 0)",
                                       [d1,d2,d3,d4,d5,d6])

for d1 in tqdm.tqdm(range(1,7)):
    for d2 in range(d1,7):
        for d3 in range(d2,7):
            for d4 in range(1,9):
                for d5 in range(d4,9):
                    for d6 in range(d5,9):
                        outcome = scipy.stats.ttest_ind([d1,d2,d3], [d4,d5,d6], equal_var=False)
                        cursor.execute(f"insert into synthetic_experiments (control_mean, control_stddev, experiment_mean, experiment_stddev, {insert_args}, was_successful) values (3.5, 1.87, 4.5, 2.45, ?,?,?,?,?,?, 1)",
                                       [d1,d2,d3,d4,d5,d6])                        
conn.commit()

cursor.execute("""create table if not exists welch_ttest_results (
  experiment_id integer not null references synthetic_experiments,
  means_differ bool,
  pvalue float)""")
cursor.execute(f"select synthetic_experiment_id, {insert_args} from synthetic_experiments left join welch_ttest_results on (synthetic_experiments.synthetic_experiment_id = welch_ttest_results.experiment_id) where welch_ttest_results.experiment_id is null")
iterator = cursor
if args.progress:
    iterator = tqdm.tqdm(cursor)
    #iterator.total = cursor.rowcount

write_cursor = conn.cursor()
for n,row in enumerate(iterator):
    control_values = []
    experiment_values = []
    for field, value in zip(['synthetic_experiment_id'] + control_columns + experiment_columns, row):
        if field.startswith('control'):
            control_values.append(value)
        elif field.startswith("experimental"):
            experiment_values.append(value)
    outcome = scipy.stats.ttest_ind(control_values, experiment_values, equal_var=False)
    write_cursor.execute("insert into welch_ttest_results (experiment_id, means_differ, pvalue) values (?,?,?)",
                   [row[0], 1 if outcome.pvalue < 0.05 else 0, outcome.pvalue])
    if n % 100 == 0:
        conn.commit()

conn.commit()
cursor.execute("""create table if not exists welch_ttest_metadata (
  type_1_error_ratio float,
  type_2_error_ratio float,
  roc_auc float,
  accuracy float
);""")

write_cursor.execute("DELETE FROM welch_ttest_metadata;")

# Retrieve the necessary data
cursor.execute("""
    SELECT s.was_successful, w.means_differ
    FROM synthetic_experiments s
    JOIN welch_ttest_results w ON s.synthetic_experiment_id = w.experiment_id;
""")
data = cursor.fetchall()

# Separate the data into true labels and predictions
true_labels = [d[0] for d in data]
predictions = [d[1] for d in data]

# Calculate the metrics
tp = sum(1 for i, j in zip(true_labels, predictions) if i == j and i)
fp = sum(1 for i, j in zip(true_labels, predictions) if j and not i)
tn = sum(1 for i, j in zip(true_labels, predictions) if not i and not j)
fn = sum(1 for i, j in zip(true_labels, predictions) if i and not j)

type_1_error_ratio = fp / (fp + tn) if fp + tn > 0 else None
type_2_error_ratio = fn / (fn + tp) if fn + tp > 0 else None
# Really, I should clean that code up and use typescorers.py

roc_auc = roc_auc_score(true_labels, predictions)
accuracy = accuracy_score(true_labels, predictions)

# Insert the new record
write_cursor.execute("""
    INSERT INTO welch_ttest_metadata (type_1_error_ratio, type_2_error_ratio, roc_auc, accuracy)
    VALUES (?, ?, ?, ?);
""", (type_1_error_ratio, type_2_error_ratio, roc_auc, accuracy))


cursor.execute("update metadata set most_recent_write = current_timestamp;")
conn.commit()
