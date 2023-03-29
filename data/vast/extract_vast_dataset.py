import argparse
import os

import pandas as pd




parser = argparse.ArgumentParser()
parser.add_argument(
	'--input_dir',
	required=True,
	help='Path to the input files. '
	     'Should contain the nf/ directory and the nf-week2.csv file '
	     'from the VAST 2013 MC3 netflow data.'
)
parser.add_argument(
	'--output_dir',
	default=None,
	help='Directory in which the dataset should be written. '
		 'If None, the current working directory is used.'
)
args = parser.parse_args()

base = args.input_dir
files = (
	*[os.path.join('nf', f'nf-chunk{i}.csv') for i in (1, 2, 3)],
	'nf-week2.csv'
)


# Extract the earliest timestamp in the dataset
start_time = 1e31
for file in files:
	df = pd.read_csv(os.path.join(base, file))
	start_time = min(start_time, df['TimeSeconds'].min())

# Extract and preprocess the flows
tup = []
for file in files:
	df = pd.read_csv(os.path.join(base, file))
	# Turn timestamps from seconds into hours since the first event
	df['timestamp'] = df['TimeSeconds'].apply(
		lambda x: int((x - start_time) // 3600)
	)
	# Select relevant flows: source and destination should be in one
	# of the two prefixes 172.0.0.0/8 (enterprise network) and
	# 10.0.0.0/8 (external hosts)
	df = df[
		(
			(
				df['firstSeenSrcIp'].str.startswith('10.')
			)|(
				df['firstSeenSrcIp'].str.startswith('172.')
			)
		)&(
			(
				df['firstSeenDestIp'].str.startswith('10.')
			)|(
				df['firstSeenDestIp'].str.startswith('172.')
			)
		)
	]
	# Group by (timestamp, source, destination)
	cnt = df.groupby(
			['timestamp', 'firstSeenSrcIp', 'firstSeenDestIp']
		)['TimeSeconds'].agg(lambda x: len(x)).reset_index()
	tup += [
		(ts, src, dst, count)
		for _, ts, src, dst, count in cnt.itertuples()
	]
df = pd.DataFrame(tup, columns=('timestamp', 'src', 'dst', 'count'))

# Save the dataset as a CSV file
fn = 'vast.csv'
if args.output_dir is None:
	fp = fn
else:
	if not os.path.exists(args.output_dir):
		os.makedirs(args.output_dir)
	fp = os.path.join(args.output_dir, fn)
df.to_csv(fp, index=False)
