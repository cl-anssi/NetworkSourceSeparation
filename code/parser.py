import argparse




def make_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'--input_dir',
		required=True,
		help='Path to the input files. '
		     'The input directory should contain three CSV files '
		     'named train.csv, validation.csv and test.csv.'
	)
	parser.add_argument(
		'--output_dir',
		default=None,
		help='Directory in which the results should be written. '
		     'If None, the current working directory is used.'
	)
	parser.add_argument(
		'--n_estimators',
		type=int,
		nargs='+',
		default=[3],
		help='Number of activity sources to infer.'
	)
	parser.add_argument(
		'--dimension',
		type=int,
		nargs='+',
		default=[10],
		help='Embedding dimension of the NMF model for each activity '
		     'source.'
	)
	parser.add_argument(
		'--lasso_coeff_W',
		type=float,
		nargs='+',
		default=[0],
		help='L1 regularization coefficient for the mixing matrix.'
	)
	parser.add_argument(
		'--l2_coeff_UV',
		type=float,
		nargs='+',
		default=[0],
		help='L2 regularization coefficient for the embedding matrices.'
	)
	parser.add_argument(
		'--epsilon',
		type=float,
		default=1e-4,
		help='Stopping criterion for the inference procedure.'
	)
	parser.add_argument(
		'--max_iter',
		type=int,
		default=200,
		help='Maximum number of iterations for the inference procedure.'
	)
	parser.add_argument(
		'--period',
		type=int,
		default=168,
		help='Period of the seasonal model used to predict the mixing '
		     'coefficients.'
	)
	parser.add_argument(
		'--verbose',
		type=int,
		default=0,
		help='Level of verbosity. '
		     'If verbose == 0, no message is displayed. '
		     'If verbose >= 1, a message is displayed at the start and '
		     'at the end of the inference procedure. '
			 'If verbose >= 2, a message is displayed after each '
			 'iteration of the inference procedure.'
	)
	parser.add_argument(
		'--device',
		default=None,
		help='Name of the device used for PyTorch tensor operations. '
		     'If None, the NumPy backend is used.'
	)
	parser.add_argument(
		'--seed',
		type=int,
		default=None,
		help='Seed of the random number generator.'
	)
	return parser