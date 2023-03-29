cd code/
for i in {0..9}; do
	for n in 2 3 4 5; do
		for d in 10 20 30 40 50; do
			python run_experiment.py --input_dir ../data/lanl/ --output_dir ../results/lanl/$n_$d/ --n_estimators $n --dimension $d --lasso_coeff_W 0 1e-5 1e-4 1e-3 --l2_coeff_UV 0 1e-5 1e-4 1e-3 --seed $i;
		done
	done
done
python compute_evaluation_metrics.py --input_dir ../results/lanl/ --output_dir ../results/