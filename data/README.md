# Datasets

This directory contains data preprocessing scripts that can be used to
reproduce our experiments using the original datasets.

### VAST dataset

The data can be found
[here](http://visualdata.wustl.edu/varepository/VAST%20Challenge%202013/challenges/MC3%20-%20Big%20Marketing/),
including the complete ground truth description of the hosts and
attacks.
We use the NetFlow data in our experiments, which are distributed
across four files: `nf/nf-chunk{1,2,3}.csv` and `nf-week2.csv`.
The script `vast/extract_vast_dataset.py` processes these files and
extracts a single file called `vast.csv`, which is the one we used
in our experiments.

### LANL dataset

The data can be found
[here](https://csr.lanl.gov/data/cyber1/).
Our experiments rely on the authentication logs (`auth.txt.gz`), and
the red team events are listed in the file `redteam.txt.gz`.
The script `lanl/extract_lanl_dataset.py` processes these files and
extracts three CSV files: `train.csv`, `validation.csv`, and `test.csv`.
These files can then be used to reproduce the experiments presented in
the paper.
