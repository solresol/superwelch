#!/usr/bin/env python3

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--control-group-size", type=int, help="Number of subjects in the control group", default=5)
parser.add_argument("--experiment-group-size", type=int, help="Number of subjects who will get the treatment", default=5)
parser.add_argument("--proportion-of-experiments-that-do-nothing", type=float,
                    help="Proportion of experiments where the experiment group and control group draw from the same pool", default=0.66)

parser.add_argument("--number-of-training-experiments", type=int, help="How much training data to create",
                    default=100000)
parser.add_argument("--number-of-testing-experiments", type=int, help="How much testing data to create",
                    default=10000)
parser.add_argument("--tag", required=True, help="A tag name for the multi-run that this run is part of")
parser.add_argument("--experimental-data-rng-seed", default=1, type=int, help="What to initialize the random number generator with to create experiments")
parser.add_argument("--rfc-seed", default=1, type=int, help="What to initialize the random number generator with for the random forest classific")
parser.add_argument("--database", default="simulation_database.sqlite", help="Where to store the data from this run")
parser.add_argument("--progress", action="store_true", help="Show a progress bar")
parser.add_argument("--summary", action="store_true", help="Show a comparative summary table")
parser.add_argument("--dry-run", action="store_true", help="Store nothing in the database")
args = parser.parse_args()

import collections
import numpy
import scipy.stats
import pandas
import sklearn.metrics
import sqlite3
import sklearn.model_selection
import sklearn.ensemble

numpy.random.seed(args.experimental_data_rng_seed)

conn = sqlite3.connect(args.database)
cursor = conn.cursor()

Experiment = collections.namedtuple("Experiment",
                                    ["ControlLoc", "ControlScale", "ControlValues",
                                   "ExperimentLoc", "ExperimentScale", "ExperimentValues",
                                   "ShouldShowResult"])
def generate_experiment():
    null_experiment = scipy.stats.uniform.rvs() <= args.proportion_of_experiments_that_do_nothing
    control_loc = scipy.stats.uniform.rvs(loc=0, scale=10)
    control_scale = scipy.stats.uniform.rvs(loc=1, scale=4)
    if null_experiment:
        experiment_loc = control_loc
        experiment_scale = control_scale
    else:
        experiment_loc = scipy.stats.uniform.rvs(loc=0, scale=10)
        experiment_scale = scipy.stats.uniform.rvs(loc=1, scale=4)
    control_values = scipy.stats.norm.rvs(loc=control_loc, scale=control_scale, size=args.control_group_size)
    experiment_values = scipy.stats.norm.rvs(loc=experiment_loc, scale=experiment_scale, size=args.experiment_group_size)
    return Experiment(ControlLoc=control_loc,
                      ControlScale=control_scale,
                     ControlValues=control_values,
                     ExperimentLoc=experiment_loc,
                     ExperimentScale=experiment_scale,
                     ExperimentValues=experiment_values,
                     ShouldShowResult=not(null_experiment))

def generate_experiments(number_of_experiments, purpose=""):
    counter_loop = range(number_of_experiments)
    if args.progress:
        import tqdm
        counter_loop = tqdm.tqdm(counter_loop)
        counter_loop.set_description(f"Generating {purpose} experiments")
    return [generate_experiment() for n in counter_loop]

def create_feature_dataframe(experiments):
    records = []
    for experiment in experiments:
        record = {}
        for i in range(args.control_group_size):
            record[f"c{i}"] = experiment.ControlValues[i]
        for i in range(args.experiment_group_size):
            record[f"x{i}"] = experiment.ExperimentValues[i]
        records.append(record)
    return pandas.DataFrame.from_records(records)

def create_target_series(experiments):
    return pandas.Series([x.ShouldShowResult for x in experiments])

training_experiments = generate_experiments(args.number_of_training_experiments, 'training')
training_targets = create_target_series(training_experiments)
training_experiments_df = create_feature_dataframe(training_experiments)

testing_experiments = generate_experiments(args.number_of_testing_experiments, 'testing')
testing_targets = create_target_series(testing_experiments)
testing_experiments_df = create_feature_dataframe(testing_experiments)

def type_1_error_ratio(y_true, y_pred):
    return sklearn.metrics.confusion_matrix(y_true, y_pred)[0][1] / len(y_true)
type_1_error_score = sklearn.metrics.make_scorer(type_1_error_ratio, greater_is_better=False)


######################################################################
# Welch. Doesn't require training.

class WelchTTest:
    def __init__(self):
        pass
    def fit(self, X,y, weights=[]):
        pass
    def predict(self, Xs):
        probs = self.predict_proba(Xs)
        return probs < 0.05
    def predict_proba(self, Xs):
        answer = []
        for experiment in Xs:
            answer.append(scipy.stats.ttest_ind(experiment.ControlValues, experiment.ExperimentValues, equal_var=False).pvalue)
        return pandas.Series(data=answer)

welch = WelchTTest()
welch_answers = welch.predict(testing_experiments)
welch_probs = welch.predict(testing_experiments)

welch_type1 = type_1_error_ratio(testing_targets, welch_answers)
welch_accuracy = sklearn.metrics.accuracy_score(testing_targets, welch_answers)
welch_roc_auc = sklearn.metrics.roc_auc_score(testing_targets, welch_probs)
confusion_matrix = sklearn.metrics.confusion_matrix(testing_targets, welch_answers)
welch_true_positives = confusion_matrix[0][0]
welch_said_success_when_actually_failed = confusion_matrix[0][1]
welch_said_failed_when_actually_succeeded = confusion_matrix[1][0]
welch_true_negatives = confusion_matrix[1][1]

rfc = sklearn.ensemble.RandomForestClassifier(random_state=args.rfc_seed, n_jobs=-1)
rfc.fit(training_experiments_df, training_targets)
rfc_answers = rfc.predict(testing_experiments_df)
rfc_probs = rfc.predict_proba(testing_experiments_df)[:, 1]

rfc_type1 = type_1_error_ratio(testing_targets, rfc_answers)
rfc_accuracy = sklearn.metrics.accuracy_score(testing_targets, rfc_answers)
rfc_roc_auc = sklearn.metrics.roc_auc_score(testing_targets, rfc_probs)
confusion_matrix = sklearn.metrics.confusion_matrix(testing_targets, rfc_answers)
rfc_true_positives = confusion_matrix[0][0]
rfc_said_success_when_actually_failed = confusion_matrix[0][1]
rfc_said_failed_when_actually_succeeded = confusion_matrix[1][0]
rfc_true_negatives = confusion_matrix[1][1]

if not args.dry_run:
    cursor.execute("""create table if not exists simulations (
    simulation_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    control_group_size int,
    experiment_group_size int,
    proportion_of_experiments_that_do_nothing float,
    number_of_training_experiments int,
    number_of_testing_experiments int,
    tag varchar,
    experimental_data_rng_seed int,
    when_run varchar default current_timestamp
);""")
    cursor.execute("insert into simulations (control_group_size, experiment_group_size, proportion_of_experiments_that_do_nothing, number_of_training_experiments, number_of_testing_experiments, tag, experimental_data_rng_seed) values (?,?,?,?,?,?,?) returning simulation_id",
                   [args.control_group_size,
                    args.experiment_group_size,
                    args.proportion_of_experiments_that_do_nothing,
                    args.number_of_training_experiments,
                    args.number_of_testing_experiments,
                    args.tag,
                    args.experimental_data_rng_seed])
    row = cursor.fetchone()
    simulation_id = row[0]

    cursor.execute("""create table if not exists welch (
    simulation_id int references simulations,
    type1 float,
    accuracy float,
    roc_auc float,
    true_positives int,
    said_success_when_actually_failed int,
    said_failed_when_actually_succeeded int,
    true_negatives int)""")
    cursor.execute("insert into welch (simulation_id, type1, accuracy, roc_auc, true_positives, said_success_when_actually_failed, said_failed_when_actually_succeeded, true_negatives) values (?,?,?,?,?,?,?,?)",
                   [simulation_id,
                    welch_type1,
                    welch_accuracy,
                    welch_roc_auc,
                    welch_true_positives,
                    welch_said_success_when_actually_failed,
                    welch_said_failed_when_actually_succeeded,
                    welch_true_negatives])

    cursor.execute("""create table if not exists random_forest (
    simulation_id int references simulations,
    n_estimators int default 100,
    criterion varchar default 'gini',
    max_depth int,
    min_samples_split int default 2,
    min_samples_leaf int default 1,
    min_weight_fraction_leaf float default 0.0,
    max_features varchar default 'sqrt',
    max_leaf_nodes int,
    max_impurity_decrease float,
    bootstrap bool default True,
    random_state int not null,
    type1 float,
    accuracy float,
    roc_auc float,
    true_positives int,
    said_success_when_actually_failed int,
    said_failed_when_actually_succeeded int,
    true_negatives int
)""")
    cursor.execute("insert into random_forest (simulation_id, random_state, type1, accuracy, roc_auc, true_positives, said_success_when_actually_failed, said_failed_when_actually_succeeded, true_negatives) values (?,?,?,?,?,?,?,?,?)",
                   [simulation_id,
                    args.rfc_seed,
                    rfc_type1,
                    rfc_accuracy,
                    rfc_roc_auc,
                    rfc_true_positives,
                    rfc_said_success_when_actually_failed,
                    rfc_said_failed_when_actually_succeeded,
                    rfc_true_negatives])
    conn.commit()




if args.summary:
    welch_type1_win = "*" if welch_type1 <= rfc_type1 else " "
    rfc_type1_win = "*" if rfc_type1 <= welch_type1 else " "
    welch_accuracy_win = "*" if welch_accuracy >= rfc_accuracy else " "
    rfc_accuracy_win = "*" if rfc_accuracy >= welch_accuracy else " "
    welch_roc_auc_win = "*" if welch_roc_auc >= rfc_roc_auc else " "
    rfc_roc_auc_win = "*" if rfc_roc_auc >= welch_roc_auc else " "
    welch_true_positives_win = "*" if welch_true_positives >= rfc_true_positives else " "
    rfc_true_positives_win = "*" if rfc_true_positives >= welch_true_positives else " "
    welch_true_negatives_win = "*" if welch_true_negatives >= rfc_true_negatives else " "
    rfc_true_negatives_win = "*" if rfc_true_negatives >= welch_true_negatives else " "
    welch_said_success_when_actually_failed_win = "*" if welch_said_success_when_actually_failed <= rfc_said_success_when_actually_failed else " "
    rfc_said_success_when_actually_failed_win = "*" if rfc_said_success_when_actually_failed <= welch_said_success_when_actually_failed else " "
    welch_said_failed_when_actually_succeeded_win = "*" if welch_said_failed_when_actually_succeeded <= rfc_said_failed_when_actually_succeeded else " "
    rfc_said_failed_when_actually_succeeded_win = "*" if rfc_said_failed_when_actually_succeeded <= welch_said_failed_when_actually_succeeded else " "
    print(f"                  |  Welch        | RandomForest")
    print(f"------------------+---------------+---------------")
    print(f" Training size    |         N/A   | {args.number_of_training_experiments:8}")
    print(f" Type 1 errors    |    { welch_type1:8.5f} {welch_type1_win} | {rfc_type1:8.5f} {rfc_type1_win}")
    print(f" Accuracy         |    { welch_accuracy:8.5f} {welch_accuracy_win} | {rfc_accuracy:8.5f} {rfc_accuracy_win}")
    print(f" ROC/AUC score    |    { welch_roc_auc:8.5f} {welch_roc_auc_win} | {rfc_roc_auc:8.5f} {rfc_roc_auc_win}")
    print(f" True positives   |    { welch_true_positives:8d} {welch_true_positives_win} | {rfc_true_positives:8d} {rfc_true_positives_win}")
    print(f" True negatives   |    { welch_true_negatives:8d} {welch_true_negatives_win} | {rfc_true_negatives:8d} {rfc_true_negatives_win}")
    print(f" Mislabel fails   |    { welch_said_success_when_actually_failed:8d} {welch_said_success_when_actually_failed_win} | {rfc_said_success_when_actually_failed:8d} {rfc_said_success_when_actually_failed_win}")
    print(f" Mislabel success |    { welch_said_failed_when_actually_succeeded:8d} {welch_said_failed_when_actually_succeeded_win} | {rfc_said_failed_when_actually_succeeded:8d} {rfc_said_failed_when_actually_succeeded_win}")

# rfc_scores = sklearn.model_selection.cross_validate(
#     sklearn.ensemble.RandomForestClassifier(),
#     experiments_df,
#     targets,
#     scoring={'accuracy': 'accuracy',
#              'type_1_error_ratio': type_1_error_score,
#              'roc_auc': 'roc_auc',
#              'confusion_matrix':
#              },
#     cv=args.cross_validation_partitions
# )
# print("Mean accuracy:", rfc_scores['test_accuracy'].mean())
# print("Mean type 1 error:", abs(rfc_scores['test_type_1_error_ratio'].mean()))
# print("Mean roc:", rfc_scores['test_roc_auc'].mean())
