import argparse
import gzip
import os

from collections import defaultdict

import pandas as pd




parser = argparse.ArgumentParser()
parser.add_argument(
	'--auth',
	required=True,
	help='Path to auth.txt.gz.'
)
parser.add_argument(
	'--redteam',
	required=True,
	help='Path to redteam.txt.gz.'
)
parser.add_argument(
	'--output_dir',
	default=None,
	help='Directory in which the dataset should be written. '
		 'If None, the current working directory is used.'
)

win_len = 3600

train_tuples = defaultdict(int)
val_tuples = defaultdict(int)
test_tuples = defaultdict(int)

# Extract and preprocess remote logons
fp = args.auth
with gzip.open(fp, 'rt') as f:
	for l in f:
		tmp = l.strip().split(',')
		if int(tmp[0]) >= 2592000:
			# We only consider the first 30 days as no red team
			# activity happens afterwards
			break
		if tmp[7] == 'LogOn' and tmp[3] != tmp[4]:
			# Consider remote logons only (source computer different
			# from destination computer)
			step = int(tmp[0]) // win_len
			tup = (step, tmp[3], tmp[4])
			if step < 168:
				# First 7 days: training set
				train_tuples[tup] += 1
			elif step < 192:
				# Eighth day: validation set
				val_tuples[tup] += 1
			else:
				# Days 9-30: test set
				test_tuples[tup] += 1

# Filter the validation and test sets: keep only edges between nodes
# which are observed at least once in the training set
nodeset = set(
		[u for _, u, _ in train_tuples]
	).union(set(
		[v for _, _, v in train_tuples]
))
test_tuples = dict([
	(t, v) for t, v in test_tuples.items()
	if t[1] in nodeset and t[2] in nodeset
])
val_tuples = dict([
	(t, v) for t, v in val_tuples.items()
	if t[1] in nodeset and t[2] in nodeset
])

# Get temporal edges corresponding to red team activity
rt = pd.read_csv(args.redteam, names=('ts', 'usr', 'src', 'dst'))
rt['win'] = rt['ts'].apply(lambda x: int(x) // win_len)
rt_tuples = set([(t.win, t.src, t.dst) for t in rt.itertuples()])

# Build dataframes
train = pd.DataFrame(
	sorted([(*t, v) for t, v in train_tuples.items()]),
	columns=('window', 'src', 'dst', 'count')
)
val = pd.DataFrame(
	sorted([(*t, v) for t, v in val_tuples.items()]),
	columns=('window', 'src', 'dst', 'count')
)
test = pd.DataFrame(
	sorted([
		(*t, v, -1 if t in rt_tuples else 1)
		for t, v in test_tuples.items()]),
	columns=('window', 'src', 'dst', 'count', 'label')
)

# Save the dataset
if args.output_dir is None:
	out_dir = ''
else:
	if not os.path.exists(args.output_dir):
		os.makedirs(args.output_dir)
	out_dir = args.output_dir
for df, name in zip(
		[train, val, test],
		['train.csv', 'validation.csv', 'test.csv']
	):
	df.to_csv(os.path.join(out_dir, name), index=False)
