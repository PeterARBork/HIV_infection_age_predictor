""" Predicts age of HIV infection from pileup files.

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
> chmod +x infection_time.py
on unix platforms). Then run the program again from the command-line
with
> python infection_time.py
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
POL_START = 2085
POL_END = 5096
LOCATIONS_TO_INCLUDE = [str(loc) for loc in list(range(POL_START, POL_END, 3))]
RESULTS_FILE = 'predicted_ages.csv'
LOG_FILE = 'age_prediction_for_hiv.log'

def main():
    """ Predicts ages of infection for all .mpileups in program's folder.

    See main script documentation.

    :return: 0 for successfully completed run, otherwise -1
    """
    logging.basicConfig(filename=LOG_FILE, level=logging.DEBUG)
    logging.info('Program started.')
    logging.info('Using intercept %.3f, slope %.3f and threshold %.3f'
                 % (INTERCEPT, SLOPE, THRESHOLD))

    relevant_mpileup_files = get_relevant_mpileup_files()
    logging.info('Read %d relevant mpileup files in this folder' % len(relevant_mpileup_files))

    logging.info('Starting calculations on each mpileup file.')
    for mpileup in relevant_mpileup_files:
        logging.info('Now working on %s.' % mpileup)
        identifier = read_identifier(mpileup)
        reference_genome = read_reference_genome(mpileup)
        (pileup_df, empty_locations,
         avg_depth, num_positions_covered) = parse_pileup(mpileup)

        if len(empty_locations) != 0:
            logging.debug('In %s, %d of the relevant locations had zero depth '
                          '(%d were covered with %.2f average depth).'
                          % (mpileup, len(empty_locations), num_positions_covered, avg_depth))

        try:
            average_hamming_distance, predicted_age = predict_age(pileup_df)
        except Exception as e:
            logging.error('File %s gave error: %s.' % (mpileup, e.args))
            continue

        try:
            jc_time = calculate_jcdistance(pileup_df)
        except Exception as e:
            logging.error('File %s gave error: %s.' % (mpileup, e.args))
            jc_time=0.0

        add_prediction_to_csv(identifier=identifier, reference=reference_genome,
                              avg_hamming=average_hamming_distance, age=predicted_age,
                              avg_depth=avg_depth, num_pos_covered=num_positions_covered,
                              jc_time=jc_time)
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

def predict_age(pileup_data: pd.DataFrame):
    """ Predicts age based on given mpileup file.

    Uses nucleotide frequency threshold hardcoded in this script.

    :param mpileup_filename: string filename (without path).
    :return: float predicted age of infection in years.
    """
    #pileup_data, empty_locations = parse_pileup(mpileup_filename)
    average_hamming = calculate_average_pairwise_distance(pileup_data)
    predicted_age = round(average_hamming * SLOPE + INTERCEPT, 1)
    
    return average_hamming, predicted_age

def calculate_jcdistance(pileup_df: pd.DataFrame) -> float:
    """ Calculates Jukes-Cantor distance estimate of number of changes.

    In the Jules-Cantor model, the frequency of mutants at a given location is
    f = (3/4) - (3/4) * exp(-mu * t),
    where mu is the rate of substituting one base for any other (including to its own, since all are equal).

    Nowak 1999 suggests an error rate such as 10**(-4) per replication and Rodrigo et al (1999) argue that the
    generation time is 1.2 days per generation. This means 10**(-4) / 1.2 errors per day, leading to mu such as
    mu = (4/3) * 10**(-4) / 1.2
    (since an error per time is (3/4) mu).

    Cuevas et al 2015 find an overall mutation rate (in plasma genomes) of (9.3 +- 2.3) * 10**(-5) per base per cell.

    Solving for the frequncy of mutatants at a given location gives the time in days
    t = -(mu)**(-1) * ln(1 - (4/3)*f)

    See https://en.wikipedia.org/wiki/Models_of_DNA_evolution#Most_common_models_of_DNA_evolution

    :param: pileup_df: pandas dataframe either with 'hamming' and 'reference' columns or which can be used in this
    module's calculate_average_pairwise_distance().
    :return: Jules-Cantor distance calculated using the hamming-based p-distance
    """
    pileup_df = pileup_df[pileup_df['reference'].isin(['A', 'C', 'T', 'G'])]
    ref_freq = lambda row: row['%s_freq' % row['reference']]
    pileup_df['ref_freq'] = pileup_df.apply(ref_freq, axis=1)

    mu = (4 / 3) * 9.3 * 10**(-5) / 1.2
    jc_t = lambda f: -(1/mu) * np.log(1 - (4 / 3) * f) / 365
    jc_filter = lambda f: jc_t(f) if jc_t(f) < np.inf else np.nan
    pileup_df['jc_t'] = pileup_df['ref_freq'].apply(jc_filter)

    avg_jc_t = pileup_df['jc_t'].mean()

    return avg_jc_t

def verify_pileup_df(pileup_df: pd.DataFrame) -> None:
    """ #TODO: STUB! Verifies pileup dataframe and throws ValueError if errors are found. """
    return None

def handheld_pileup_parsing(pileup_filename: str) -> pd.DataFrame:
    """ Parses mpileup carefully to avoid errors when first line has empty columns. """

    def line_to_record(line: str) -> dict:
        values = line.strip('\n').split('\t')
        if len(values) != 6:
            logging.warning('%d values found in line: %s.' % (len(values), line))
            return dict()

        col_names = ['disregard', 'location', 'reference', 'depth', 'nucleotides', 'qualities']
        return {col_name: value for col_name, value in zip(col_names, values)}

    def location_filter(record: dict):
        location_criteria = record.get('location', 'missing') in LOCATIONS_TO_INCLUDE
        content_criteria = len(record.get('nucleotides', '')) > 0 and len(record.get('qualities', '')) > 0
        return True if location_criteria and content_criteria else False

    with open(pileup_filename, 'r') as f:
        contents = list(filter(location_filter, map(line_to_record, f)))

    if len(contents) == 0:
        logging.warning('No nuclotide locations match criteria in %s.' % pileup_filename)

    return pd.DataFrame(contents)

def parse_pileup(pileup_filename: str) -> (pd.DataFrame, pd.Series):
    """ Returns dataframe with frequencies from pileup file.

    :param pileup_filename: string filename.
    :return: tuple of dataframe of locations, dataframe of empty locations, average depth, and number of positions
    covered.
    """

    df = handheld_pileup_parsing(pileup_filename)
    df = df[['location', 'reference', 'depth', 'nucleotides']]

    verify_pileup_df(df)

    logging.debug('Out of (%d - %d) / %d = %d  expected bases in pol, we have %d bases covered in the pileup.'
                  % (POL_END, POL_START, 3, (POL_END - POL_START) / 3, sum(df.location.isin(LOCATIONS_TO_INCLUDE ))))

    # keep only 3rd base in codons
    df = df[df.location.isin(LOCATIONS_TO_INCLUDE)]
    empty_locations = list(set(LOCATIONS_TO_INCLUDE) - set(df.location.unique()))
    df = df.dropna(subset=['nucleotides'])

    # Keep only valid reference letters.
    bad_ref_locations = df.location[~df.reference.isin(['A', 'C', 'G', 'T', 'a', 'c', 'g', 't'])]
    if len(bad_ref_locations) > 0:
        logging.warning('Pileup file %s has %d locations with bad reference, which will be dropped (%s).'
                        % (pileup_filename, len(bad_ref_locations), ', '.join(list(bad_ref_locations))))
        df = df[df.reference.notnull()]


    def pileup_bases_to_clean_bases(pileup_df_row: pd.Series) -> str:
        reference = pileup_df_row['reference'].upper()
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

    try:
        df['tidy_bases'] = df.apply(pileup_bases_to_clean_bases, axis=1)
    except ValueError as ve:
        logging.error('Could not tidy bases. The dataframe read has the following columns:\n%s\n'
                      'The first five rows looks like this:\n%s'
                      % (df.columns, df.head()))
        raise ve
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

    df = df.dropna()
    avg_depth = pd.to_numeric(df.depth).mean()
    num_positions_covered = len(df)

    return df, empty_locations, avg_depth, num_positions_covered

def calculate_average_pairwise_distance(parsed_pileup_df: pd.DataFrame) -> float:
    """ Predicts average pairwise distance on dataframe from pileup file.

    Using Puller, Neher Albert (2017) which says that this is the conventional Nei-Li nucleotide diversity.

    In the pre-print, Puller, Neher and Albert (2017) called this the average hamming distance, so there was a bit of
    confusion and the code still reflects that.

    :param parsed_pileup_df: pandas DataFrame with nucleotide frequencies.
    :return: float predicate age in years.
    """
    def calculate_summand(pileup_row: pd.Series):
        frequencies = [pileup_row['A_freq'], pileup_row['T_freq'], pileup_row['C_freq'], pileup_row['G_freq']]

        tofte_score_factor = sum([freq * (1 - freq) for freq in frequencies])

        # alternatively use the frequency for the nucleotide pileup_row['reference']
        x_m = max(frequencies)

        heaviside_factor = 1 if THRESHOLD < (1 - x_m) else 0

        hamming_distance = heaviside_factor * tofte_score_factor

        return hamming_distance

    parsed_pileup_df['hamming'] = parsed_pileup_df.apply(calculate_summand, axis=1)

    average_hamming_distance = sum(parsed_pileup_df['hamming']) / len(parsed_pileup_df)

    return average_hamming_distance

def add_prediction_to_csv(*args, **kwargs):
    """ Adds prediction to predicted_ages.csv in same folder as this script.

    :param identifier: string identifier for mpileup file
    :param reference_genome: string name of reference genome for mpileup file.
    :return: None

    identifier: str, reference_genome: str,
                          average_hamming_distance: float, predicted_age: float,
                          avg_depth, num_positions_covered,

    """
    if kwargs:
        assert not args, 'Do not give both key-worded values and values without keywords to be saved.'
        cols = sorted(kwargs.keys())
        values = [str(kwargs[key]) for key in cols]
        if not Path(RESULTS_FILE).is_file():
            with open(RESULTS_FILE, mode='w') as f:
                f.write(','.join(cols) + '\n')
    elif args:
        values = [str(arg) for arg in args]
    else:
        raise ValueError('Either args or kwargs must be provided to add_prediction_to_csv.')

    logging.info('Writing line to csv: %s' % ','.join(values))
    with open(RESULTS_FILE, mode='a') as f:
        observation = ','.join(values)
        f.write(observation + '\n')

    return None

if __name__ == '__main__':
    main()
