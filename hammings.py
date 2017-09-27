""" Hammings predicts age of HIV infection from pileup files.

All files in same folder as this executable with pileup filenames
will be used as input if their names have the following structure:
XXX-XXXX-X_cleaned_map_ZZZZZ.mpileup,
where X's are digits 0-9 (to identify sample) and Z's are variable
length strings for HIV reference genome identification.

The predicted ages, along with the digit identifiers and reference
string names will be added to a csv file in the same folder as this
executable named
predicted_ages.csv
(This file will be created if it does not already exist).

The infection age prediction will be made according to
Puller, Neher Alber (in press, see bioRxiv):
http://www.biorxiv.org/content/early/2017/04/21/129387

Usage example:
Place this executable in a folder with only .mpileup files to be
calculated on and make sure the file is executable (i.e. use
> chmod +x hammings.py
on unix platforms). Then run the program again from the command-line
with
> python hammings.py
This program is in python version 3, so make sure to use python 3.
"""
import logging
from pathlib import Path
from os import listdir
import pandas as pd
import numpy as np

SLOPE = 250.28
INTERCEPT = -0.08
THRESHOLD = 0.0

def main():
    """ Predicts ages of infection for all .mpileups in program's folder.

    See main script documentation.

    :return: 0 for successfully completed run, otherwise -1
    """
    logging.basicConfig(filename='hammings.log', level=logging.DEBUG)
    logging.info('Program started.')
    logging.info('Using intercept %.3f, slope %.3f and threshold %.3f'
                 % (INTERCEPT, SLOPE, THRESHOLD))

    relevant_mpileup_files = get_relevant_mpileup_files()
    logging.info('Read %d relevant mpileup files in this folder' % len(relevant_mpileup_files))

    logging.info('Starting calculations on each mpileup file.')
    for mpileup in relevant_mpileup_files:
        identifier = read_identifier(mpileup)
        reference_genome = read_reference_genome(mpileup)
        try:
            average_hamming_distance, predicted_age = predict_age(mpileup)
        except Exception as e:
            logging.error('File %s gave error.' %mpileup)
            raise e

        add_prediction_to_csv(identifier, reference_genome,
                              average_hamming_distance, predicted_age)
        logging.info('Predicted age %s from Hamming distance %.3f for identifier %s (reference %s).'
                     % (predicted_age, average_hamming_distance, identifier, reference_genome))

    logging.info('Completed all calculations, now finishing in good form.')
    return 0

def get_relevant_mpileup_files():
    """ Reads this folder for mpileup files and filters them.

    :return: list of string filenames (without path).
    """
    current_path = str(Path.cwd())
    all_files = listdir(current_path)
    relevant_files = list(filter(is_filename_relevant, all_files))
    return relevant_files

def is_filename_relevant(filename: str):
    """ Returns True if filename looks like its a good mpileup file.

    A good filename looks like described in the top docstring.

    :param filename: string filename without path
    :return: True if filename if it looks right.
    """
    # XXX-XXXX-X_cleaned_map_ZZZZZ.mpileup
    dash_split = filename.split('-')
    if len(dash_split) != 3:
        return False
    elif filename.split('.')[1] != 'mpileup':
        return False
    elif '_cleaned_map_' not in filename:
        return False
    else:
        return True

def read_identifier(mpileup_filename: str):
    """ Returns identifer from filename.

    :param mpileup_filename: string filename (without path).
    :return: string identifier (XXX-XXXX-X, each X being a digit)
    """
    first_two = mpileup_filename.split('-')[0:2]
    third = mpileup_filename.split('-')[2][0]

    return '-'.join(first_two) + '-' + third

def read_reference_genome(mpileup_filename: str):
    """ Returns reference genome from filename.

    :param mpileup_filename: string filename (without path).
    :return: string reference genome name.
    """
    after_last_underscore = mpileup_filename.split('_')[-1]
    reference = after_last_underscore.split('.')[0]
    return reference

def predict_age(mpileup_filename: str):
    """ Predicts age based on given mpileup file.

    Uses nucleotide frequency threshold hardcoded in this script.

    :param mpileup_filename: string filename (without path).
    :return: float predicted age of infection in years.
    """
    pileup_data, empty_locations = parse_pileup(mpileup_filename)
    average_hamming = calculate_average_hamming_distance(pileup_data)
    predicted_age = round(average_hamming * SLOPE + INTERCEPT, 1)

    if len(empty_locations) != 0:
        logging.debug('In %s, %d of the relevant locations had zero depth.'
                      % (mpileup_filename, len(empty_locations)))
    
    return average_hamming, predicted_age

def parse_pileup(pileup_filename: str) -> pd.DataFrame:
    """ Returns dataframe with frequencies from pileup file.

    :param pileup_filename: string filename.
    :return: pandas dataframe.
    """
    pol_start = 2085
    pol_end = 5096
    locations_to_include = [str(loc) for loc in list(range(pol_start, pol_end, 3))]

    # Rows in pandas are likely 0-indexed whereas the first row in the file is location 1 in the genome
    df = pd.read_csv(pileup_filename, delim_whitespace=True, header=None, usecols=[1, 2, 3, 4],
                     names=['location', 'reference', 'depth', 'nucleotides'], dtype=object)  # , index_col=0)

    logging.debug('Out of (%d - %d) / %d = %d  expected bases in pol, we have %d bases covered in the pileup.'
                  % (pol_end, pol_start, 3, (pol_end - pol_start) / 3, sum(df.location.isin(locations_to_include))))

    # keep only 3rd base in codons
    df = df[df.location.isin(locations_to_include)]
    empty_locations = df.location[df.nucleotides == np.nan]
    df = df.dropna(subset=['nucleotides'])

    def pileup_bases_to_clean_bases(pileup_df_row: pd.Series) -> str:
        reference = pileup_df_row['reference']
        if not type(reference) is str:
            raise ValueError('location %d has non-string reference.' % pileup_df_row.location)
        pileup_nucleotides = {',': reference,
                              '.': reference,
                              'a': 'A', 'A': 'A',
                              'c': 'C', 'C': 'C',
                              't': 'T', 'T': 'T',
                              'g': 'G', 'G': 'G'}
        result = ''
        for letter in pileup_df_row.nucleotides:
            result += pileup_nucleotides.get(letter, '')

        return result

    df['tidy_bases'] = df.apply(pileup_bases_to_clean_bases, axis=1)
    df = df.drop('nucleotides', axis=1)


    def calculate_nucleotide_frequency(pileup_df_row, nucleotide: str) -> float:
        num_occurences = pileup_df_row.tidy_bases.count(nucleotide)
        depth = len(pileup_df_row.tidy_bases)
        try:
            freq = num_occurences / depth
            return freq
        except ZeroDivisionError as zde:
            logging.warning('Got ZeroDivisionError on location %s' % pileup_df_row['location'])
            return None

    for nucleotide in ['A', 'C', 'G', 'T']:
        df['%s_freq' % nucleotide] = df.apply(calculate_nucleotide_frequency, axis=1, nucleotide=nucleotide)

    df['freqs_sum'] = df.apply(lambda row: row['A_freq'] + row['C_freq'] + row['G_freq'] + row['T_freq'], axis=1)

    percent_with_wrong_frequency_sum = len(df[abs(df['freqs_sum'] - 1) > 0.01]) / len(df) * 100
    if percent_with_wrong_frequency_sum > 1:
        logging.debug('There are %d%s rows where the frequencies do not sum to one, perhaps because of rounding.'
                      % (percent_with_wrong_frequency_sum, '%'))

    return df, empty_locations

def calculate_average_hamming_distance(parsed_pileup_df: pd.DataFrame) -> float:
    """ Predicts Hamming distance on dataframe from pileup file.

    :param parsed_pileup_df: pandas DataFrame with nucleotide frequencies.
    :return: float predicate age in years.
    """
    def calculate_summand(pileup_row: pd.Series):
        frequencies = [pileup_row['A_freq'], pileup_row['T_freq'], pileup_row['C_freq'], pileup_row['G_freq']]

        tofte_score_factor = sum([freq * (1 - freq) for freq in frequencies])

        # alternatively use the frequency for the nucleotide pileup_row['reference']
        x_m = max(frequencies)

        heaviside_factor = 1 if THRESHOLD < (1 - x_m) else 0

        hammond_distance = heaviside_factor * tofte_score_factor

        return hammond_distance

    parsed_pileup_df['hammonds'] = parsed_pileup_df.apply(calculate_summand, axis=1)

    average_hammond_distance = sum(parsed_pileup_df['hammonds']) / len(parsed_pileup_df)

    return average_hammond_distance

def add_prediction_to_csv(identifier: str, reference_genome: str,
                          average_hamming_distance: float, predicted_age: float):
    """ Adds prediction to predicted_ages.csv in same folder as this script.

    :param identifier: string identifier for mpileup file
    :param reference_genome: string name of reference genome for mpileup file.
    :return: None
    """
    with open('predicted_ages.csv', mode='a') as f:
        observation = ','.join([identifier, reference_genome,
                                str(average_hamming_distance), str(predicted_age)])
        f.write(observation + '\n')

    return None

if __name__ == '__main__':
    main()
