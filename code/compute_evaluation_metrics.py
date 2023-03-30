import argparse
import gzip
import os

import numpy as np
import sklearn.metrics as met

from joblib import Parallel, delayed




def read_snmf_results(dir_path, num_interp_points=10000, jobs=10):
    '''
    Reads the raw results output by the run_experiment.py script
    and computes evaluation metrics (ROC curve, NDCG@1%).


    Arguments
    ---------
    dir_path : str
        Path to the raw results.
        The expected structure of the directory is as follows:
        - for each evaluated hyperparameter pair
          (n_estimators, embedding_dim), there is a subdirectory
          called {n_estimators}_{embedding_dim}
        - each of these subdirectories should contain the compressed
          result files for different random seeds
    num_interp_points : int, default=10000
        Number of points used to interpolate the ROC curve.
    jobs : int, default=10
        Number of processes to create for parallelization.

    Returns
    -------
    res : dict
        Evaluation metrics for each task and each hyperparameter pair.
    xs : array of shape (num_interp_points,)
        Coordinates used for ROC curve interpolation.

    '''

    xs = np.linspace(0, 1, num_interp_points)
    dirs = os.listdir(dir_path)
    res = {}
    files = [
        (dn, fn)
        for dn in dirs
        for fn in os.listdir(os.path.join(dir_path, dn))
    ]
    seeds = [
        int(fname.split('_')[-1].split('.')[0])
        for _, fname in files
    ]
    n_seeds = max(seeds) + 1
    def read_file(dname, fname):
        # Define a subfunction to enable parallel computing
        dim, n_est = int(dname.split('_')[1]), int(dname.split('_')[2])
        seed = int(fname.split('_')[-1].split('.')[0])
        fp = os.path.join(dir_path, dname, fname)
        print(fp)
        roc = {
            task: []
            for task in ('malicious', 'easy', 'historical', 'inductive')
        }
        with gzip.open(fp, 'rt') as f:
            dat = json.loads(f.read())
            for k in roc.keys():
                y_true = np.array(dat[k]['labels'])
                y_pred = np.array(dat[k]['scores'])
                if k == 'malicious':
                    ndcg = met.ndcg_score(
                        -y_true[np.newaxis, :] + 1,
                        -y_pred[np.newaxis, :],
                        k=len(y_true) // 100
                    )
                    y_true, y_pred = -y_true, -y_pred
                fpr, tpr, _ = met.roc_curve(y_true, y_pred)
                ys = np.interp(xs, fpr, tpr)
                roc[k] = ys
            val_score = dat['val_score']
        return (dim, n_est, seed), (roc, ndcg, val_score)
    tmp = Parallel(n_jobs=jobs)(
        delayed(read_file)(dname, fname)
        for dname, fname in files
    )
    for (d, n, s), (roc, ndcg, val_score) in tmp:
        if (d, n) not in res:
            res[(d, n)] = {
                'roc': {},
                'ndcg': [0] * n_seeds,
                'val_score': [0] * n_seeds
            }
        for task in roc:
            if task not in res[(d, n)]['roc']:
                res[(d, n)]['roc'][task] = [None] * n_seeds
            res[(d, n)]['roc'][task][s] = roc[task]
        res[(d, n)]['val_score'][s] = val_score
        res[(d, n)]['ndcg'][s] = ndcg
    return res, xs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_dir',
        required=True,
        help='Path to the raw results. '
             'The expected structure of the directory is as follows: '
             '(i) for each evaluated hyperparameter pair '
             '(n_estimators, embedding_dim), there is a subdirectory '
             'called {n_estimators}_{embedding_dim}; '
             '(ii) each of these subdirectories should contain the '
             'compressed result files for different random seeds'
    )
    parser.add_argument(
        '--output_dir',
        default=None,
        help='Directory where the results should be written. '
             'If None, the current working directory is used.'
    )
    parser.add_argument(
        '--n_interpolation_points',
        type=int,
        default=1e4,
        help='Number of points used to interpolate the ROC curve.'
    )
    parser.add_argument(
        '--n_jobs',
        type=int,
        default=10,
        help='Number of processes to create for parallelization.'
    )
    args = parser.parse_args()

    # Read raw results and compute evaluation metrics
    res, xs = read_snmf_results(
        args.input_dir,
        num_interp_points=args.n_interpolation_points,
        jobs=args.n_jobs
    )
    # Write outputs
    to_write = {
        'xs': xs.tolist()
    }
    for dim, n_est in res:
        model = f'{dim}_{n_est}'
        to_write[model] = {
            'roc': {
                task: [
                    y.tolist()
                    for y in res[(dim, n_est)]['roc'][task]
                ]
                for task in res[(dim, n_est)]['roc']
            },
            'val_score': res[(dim, n_est)]['val_score'],
            'ndcg': res[(dim, n_est)]['ndcg']
        }
    fn = f'results_lanl.json.gz'
    if args.output_dir is None:
        fp = fn
    else:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        fp = os.path.join(args.output_dir, fn)
    with gzip.open(fp, 'wt') as out:
        out.write(json.dumps(to_write))
