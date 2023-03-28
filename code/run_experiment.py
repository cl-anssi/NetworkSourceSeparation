import gzip
import json
import os

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score

from snmf import SuperposedNMF, TorchSuperposedNMF
from edge_generator import EdgeGenerator
from parser import make_parser




parser = make_parser()
args = parser.parse_args()

######################
# Data preprocessing #
######################

train = pd.read_csv(os.path.join(args.input_dir, 'train.csv'))
validation = pd.read_csv(os.path.join(args.input_dir, 'validation.csv'))
test = pd.read_csv(os.path.join(args.input_dir, 'test.csv'))

nodelist = sorted(list(set().union(*[
	set(df['src']).union(set(df['dst']))
	for df in (train, validation, test)
])))

X_train = train.to_numpy()

###############################################
# Training of one estimator for each possible #
# combination of hyperparameters              #
###############################################

if args.device is None:
	estimators = [
		SuperposedNMF(
			n_estimators=n,
			dimension=dimension // n,
			l2_coeff_UV=l2_coeff_UV,
			lasso_coeff_W=lasso_coeff_W,
			epsilon=args.epsilon,
			max_iter=args.max_iter,
			verbose=args.verbose,
			random_state=args.seed
		).fit(X_train, nodelist=nodelist)
		for n in args.n_estimators
		for dimension in args.dimension
		for l2_coeff_UV in args.l2_coeff_UV
		for lasso_coeff_W in args.lasso_coeff_W
	]
else:
	estimators = [
		TorchSuperposedNMF(
			n_estimators=n,
			dimension=dimension // n,
			l2_coeff_UV=l2_coeff_UV,
			lasso_coeff_W=lasso_coeff_W,
			epsilon=args.epsilon,
			max_iter=args.max_iter,
			verbose=args.verbose,
			device=args.device,
			random_state=args.seed
		).fit(X_train, nodelist=nodelist)
		for n in args.n_estimators
		for dimension in args.dimension
		for l2_coeff_UV in args.l2_coeff_UV
		for lasso_coeff_W in args.lasso_coeff_W
	]

######################################################
# Evaluation of each estimator on the validation set #
# and selection of the best model                    #
######################################################

X_val = validation.to_numpy()

# Generate negative edges
generator = EdgeGenerator(
	X_train,
	np.concatenate([X_val, np.ones((X_val.shape[0], 1))], axis=1),
	random_state=args.seed
)
scores = [[] for _ in estimators]
for data_tuple in zip(
		generator.generate_dataset('easy'),
		generator.generate_dataset('historical'),
		generator.generate_dataset('inductive')
	):
	for data in data_tuple:
		X_task, y_task = data[:, :-1], data[:, -1].astype(int)
		for i, est in enumerate(estimators):
			# Compute anomaly scores
			y_pred = est.score_samples(
				X_task,
				period=args.period
			)
			scores[i].append(roc_auc_score(y_task, y_pred))
	for est in estimators:
		# Infer the mixing coefficients for the current time step
		est.fit_and_append_weights(X_task[y_task == 1, :])
idx = np.argmax([sum(s) for s in scores])
est = estimators[idx]

#######################################################
# Retraining on the whole training set (including the #
# held-out validation data)                           #
#######################################################

X_train = np.concatenate([X_train, X_val], axis=0)
est.fit(X_train, nodelist=nodelist)

##############################
# Evaluation on the test set #
##############################

res = {
	task: {'scores': [], 'labels': []}
	for task in ('malicious', 'easy', 'historical', 'inductive')
}
res['val_score'] = scores[idx]

X_test = test.to_numpy()

# Generate negative edges
generator = EdgeGenerator(
	X_train,
	X_test[X_test[:, -1] != -1],
	random_state=args.seed
)
idx = test.groupby(['window']).groups
ben_mal = [
	test.loc[idx[t], :].to_numpy()
	for t in sorted(list(idx.keys()))
]
for window, (X_mal, X_easy, X_hist, X_induct) in enumerate(zip(
		ben_mal,
		generator.generate_dataset('easy'),
		generator.generate_dataset('historical'),
		generator.generate_dataset('inductive')
	)):
	print(f'Reading window {window}...')

	for task, X in zip(
			['malicious', 'easy', 'historical', 'inductive'],
			[X_mal, X_easy, X_hist, X_induct]
		):
		# Compute anomaly scores
		y = est.score_samples(
			X[:, :-1],
			period=args.period
		)
		lab = X[:, -1].astype(int)
		res[task]['scores'] += y.tolist()
		res[task]['labels'] += lab.tolist()
	# Infer the mixing coefficients for the current time step
	est.fit_and_append_weights(X_mal[:, :-1])

#################
# Write results #
#################

fn = f'results_{args.model}_{args.seed}.json.gz'
if args.output_dir is None:
	fp = fn
else:
	if not os.path.exists(args.output_dir):
		os.makedirs(args.output_dir)
	fp = os.path.join(args.output_dir, fn)
with gzip.open(fp, 'wt') as out:
	out.write(json.dumps(res))
