#!/usr/bin/env python3

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model-database", required=True, help="SQLite database where train/bmap3.py ran")
parser.add_argument("--session-database", required=True, help="Database where we will store the values that users enter (probably via the steam deck)")
import sqlite3
import sys
from io import BytesIO
import joblib
import re
import scipy
import pandas

args = parser.parse_args()

model_conn = sqlite3.connect(args.model_database)
cursor = model_conn.cursor()
cursor.execute("SELECT model FROM bmap3_storage")
row = cursor.fetchone()
if row is None:
    sys.exit(f"{args.model_database} does not contain a trained BMAP3 model")

model_buffer = BytesIO(row[0])
model = joblib.load(model_buffer)
# Now `model` is the deserialized scikit-learn pipeline
cursor.execute("select control_group_size, experiment_group_size from metadata")
row = cursor.fetchone()
if row is None:
    sys.exit(f"Metadata in {args.sqlite_database} has not been initialized correctly")
control_group_size, experiment_group_size = row

control_columns = [f"control{i+1}" for i in range(control_group_size)]
experiment_columns = [f"experimental{i+1}" for i in range(experiment_group_size)]
select_args = ", ".join(control_columns + experiment_columns)
create_defs = [f"{x} float" for x in (control_columns + experiment_columns)]
columns = "\n, ".join(create_defs)

word_completions = ['new', 'test'] + [f"{x}=" for x in (control_columns + experiment_columns) ]

model_conn.close()


conn = sqlite3.connect(args.session_database)
cursor = conn.cursor()
cursor.execute("pragma foreign_keys = on;")
cursor.execute(f"""create table if not exists real_experiments (
  real_experiment_id integer primary key,
  experiment_started timestamp default current_timestamp,
  {columns},
  experiment_evaluated timestamp default current_timestamp,
  welch_ttest_result bool,
  welch_ttest_pvalue float,
  welch_ttest_probability float,
  bmap3_result bool,
  bmap3_pvalue float,
  ground_truth bool
)""")

cursor.execute(f"""create table if not exists current_experiment (
  real_experiment_id integer references real_experiments
)""")


if sys.stdin.isatty():
    from prompt_toolkit import PromptSession, print_formatted_text
    from prompt_toolkit.formatted_text import FormattedText
    from prompt_toolkit.completion import WordCompleter
    session = PromptSession()
    word_completer = WordCompleter(word_completions)
else:
    print_formatted_text = print
    FormattedText = lambda x: "".join([t[1] for t in x])




all_input_consumed = False

cursor.execute("select real_experiment_id from current_experiment")
row = cursor.fetchone()
if row is None:
    current_experiment_id = None
else:
    current_experiment_id = row[0]

while True:
    if all_input_consumed:
        break
    try:
        if sys.stdin.isatty():
            if current_experiment_id is None:
                prompt = "BMAP3-cli (no experiment): "
                c = WordCompleter(['new'])
            else:
                prompt = f"BMAP3-cli #{current_experiment_id}: "
                c = word_completer
            user_input = session.prompt(prompt, completer=c)
        else:
            user_input = sys.stdin.read()
            all_input_consumed = True
    except EOFError:
        break

    if user_input.strip() == '':
        continue
    # Crappy command interpreter.
    if user_input.lower().strip().startswith("new"):
        print_formatted_text(FormattedText([('fg:blue', "Creating a new experiment...")]), end='')
        cursor.execute("insert into real_experiments (experiment_started) values (current_timestamp) returning real_experiment_id")
        row = cursor.fetchone()
        current_experiment_id = row[0]
        cursor.execute("delete from current_experiment;")
        cursor.execute("insert into current_experiment (real_experiment_id) values (?)", [current_experiment_id])
        conn.commit()
        print_formatted_text(FormattedText([('fg:blue', f" experiment ID = {current_experiment_id}...")]))
        continue

    if current_experiment_id is not None and '=' in user_input:
        l, r = user_input.split('=', 1)
        l = l.strip().lower()
        r = r.strip().lower()
        if l.startswith('control') or l.startswith('experimental'):
            try:
                if l.startswith('control'):
                    which_var = l[len('control'):]
                    which_var = int(which_var)
                    if which_var > control_group_size + 1:
                        print_formatted_text(FormattedText([('fg:red', f"There are only {control_group_size} controls")]))
                        continue
                if l.startswith('experimental'):
                    which_var = l[len('experimental'):]
                    which_var = int(which_var)
                    if which_var > experiment_group_size + 1:
                        print_formatted_text(FormattedText([('fg:red', f"There are only {experiment_group_size} experimental values")]))
                        continue
            except ValueError:
                print_formatted_text(FormattedText([('fg:red', f"Column {l} was not understood")]))
                continue
            try:
                set_value = float(r)
            except ValueError:
                print_formatted_text(FormattedText([('fg:red', f"Value {r} was not understood")]))
                continue
            cursor.execute(f"update real_experiments set {l} = ? where real_experiment_id = ?",
                           [set_value, current_experiment_id])
            conn.commit()
            print_formatted_text(FormattedText([('fg:blue', f"{l} has been set to be {set_value}.")]))
            continue

    if current_experiment_id is not None and user_input.lower().strip().startswith("test"):
        cursor.execute(f"select {select_args} from real_experiments where real_experiment_id = ?",
                      [current_experiment_id])
        row = cursor.fetchone()
        if row is None:
            sys.exit("Database integrity error")
            
        control_values = []
        experiment_values = []
        bmap_df = pandas.DataFrame()
        for field, value in zip(control_columns + experiment_columns, row):
            if field.startswith('control'):
                control_values.append(value)
                bmap_df[field] = [value]
            elif field.startswith("experimental"):
                experiment_values.append(value)
                bmap_df[field] = [value]                
        welch_outcome = scipy.stats.ttest_ind(control_values, experiment_values, equal_var=False)
        welch_pretty_result = "experiment succeeded" if welch_outcome.pvalue < 0.05 else "experiment failed"
        welch_result = 1 if welch_outcome.pvalue < 0.05 else 0
        print_formatted_text(FormattedText([('fg:green', f"Welch t-test reported: {welch_pretty_result} with p-value {welch_outcome.pvalue}")]))
        prediction = model.predict(bmap_df)[0]
        probs = model.predict_proba(bmap_df)[0]
        model_pretty_result = "experiment succeeded" if prediction == 1 else "experiment failed"
        print_formatted_text(FormattedText([('fg:green', f"BMAP3 test reported: {model_pretty_result} with probability of it being successful = {probs[1]}")]))
        continue
    print_formatted_text(FormattedText([('fg:red', f"Command {user_input} was not understood")]))
    #print_formatted_text(FormattedText([('bold',user_input)]))
    # print_formatted_text(FormattedText([('fg:black',"Processing end reason: "), ('fg:yellow', finish_reason)]))
    # print_formatted_text(FormattedText([('fg:yellow','Thinking...')]), end='\r')

    # print_formatted_text(FormattedText([('fg:red', error_message)]))
    # print_formatted_text(FormattedText([('fg:green', answer)]))
