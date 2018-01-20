# HIV_infection_age_predictor
Uses folder of next generation sequencing files (.mpileups) to predict ages for each sample's HIV infection.

Implements the prediction method developed by 
Puller, Neher Alber (in press, see bioRxiv):
http://www.biorxiv.org/content/early/2017/04/21/129387

Furthermore implements a method described in the accompanying jupyter notebook (under development).

Assumes Python version >= 3.5

# Usage
Place hammings.py in folder with .mpileup files and execute from command-line.

Optionally set intercept, slope and threshold in the script by hard-coding your settings in the first few lines.
