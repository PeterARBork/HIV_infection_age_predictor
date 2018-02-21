""" Utilities for HIV time since infection calculations. """

import logging
from pathlib import Path
from os import listdir
import pandas as pd
from typing import List

RESULTS_FILE = 'predicted_ages.csv'
LOG_FILE = 'age_prediction_for_hiv.log'

def count_depths():
    """ Counts depths """
    relevant_mpileup_files = get_relevant_mpileup_files()
    relevant_mpileup_files = relevant_mpileup_files[:50] if len(relevant_mpileup_files) >= 50 else relevant_mpileup_files
    for mpileup in relevant_mpileup_files:
        line_dicts = parse_pileup_to_contig(mpileup, POL_LOCATIONS)
        try:
            pos_depths = [(d['location'], d['depth']) for d in line_dicts]
        except TypeError as te:
            logging.error('In pileup %s we got TypeError: %s' % (mpileup, te.args))
            continue
        pos_dicts = {pos: depth for (pos, depth) in pos_depths}
        pos_dicts['AA_filename'] = mpileup
        add_prediction_to_csv(**pos_dicts)

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
    return True if 'pileup' in filename else False
    #dash_split = filename.split('-')
    #if len(dash_split) != 3:
    #    return False
    #elif filename.split('.')[1] != 'mpileup':
    #    return False
    #elif '_cleaned_map_' not in filename:
    #    return False
    #else:
    #    return True

def read_identifier(mpileup_filename: str):
    """ Returns identifer from filename.
    """
    if len(mpileup_filename.split()) > 2:
        first_two = mpileup_filename.split('-')[0:2]
        third = mpileup_filename.split('-')[2][0]
        return mpileup_filename.split('.')[0]
    else:
        return mpileup_filename.split('.')[0]

def read_reference_genome(mpileup_filename: str):
    """ Returns reference genome from filename.

    :param mpileup_filename: string filename (without path).
    :return: string reference genome name.
    """
    after_last_underscore = mpileup_filename.split('_')[-1]
    reference = after_last_underscore.split('.')[0]
    return reference

def parse_pileup_to_contig(pileup_filename: str, start: int, end: int) -> List[dict]:
    """ Parses mpileup carefully to avoid errors when first line has empty columns. """

    def add_tidy_bases(pileup_row: dict) -> dict:
        reference = pileup_row['reference'].upper()
        pileup_nucleotides = {',': reference, '.': reference,
                              'a': 'A', 'A': 'A',
                              'c': 'C', 'C': 'C',
                              't': 'T', 'T': 'T',
                              'g': 'G', 'G': 'G'}
        result = ''
        for letter in pileup_row['nucleotides']:
            result += pileup_nucleotides.get(letter, '')

        pileup_row.update(tidy_bases=result)
        return pileup_row

    def add_frequencies(row: dict) -> dict:
        for nucleotide in ['A', 'C', 'T', 'G']:
            effective_depth = len(row['tidy_bases'])
            if effective_depth > 0.0:
                row['%s_freq' % nucleotide] = row['tidy_bases'].count(nucleotide) / effective_depth
            else:
                row['%s_freq' % nucleotide] = 0.0

        row['freq_sum'] = row['A_freq'] + row['C_freq'] + row['T_freq'] + row['G_freq']
        return row

    def line_to_record(line: str) -> dict:
        values = line.strip().split('\t')
        if len(values) != 6:
            #logging.debug('%d values found in line: %s.' % (len(values), line.strip()))
            return dict()

        col_names = ['disregard', 'location', 'reference', 'depth', 'nucleotides', 'qualities']
        result_d = {col_name: value for col_name, value in zip(col_names, values)}

        result_d['location'] = int(result_d['location'])
        result_d = add_tidy_bases(result_d)

        result_d = {key: value for key, value in result_d.items()
                    if key in ['location', 'reference', 'tidy_bases']}

        result_d = add_frequencies(result_d)
        return result_d

    def location_sanity_filter(record: dict) -> bool:
        loc = int(record.get('location', -1))
        location_criteria = True if (loc - 1) >= start and (loc - 1) <= end else False
        got_ref = True if record.get('reference', '').upper() in ['A', 'C', 'T', 'G'] else False
        got_bases = True if len(record.get('tidy_bases', '')) > 0 else False
        return True if location_criteria and got_ref and got_bases else False

    with open(pileup_filename, 'r') as f:
        contents = list(filter(location_sanity_filter, map(line_to_record, f)))

    num_freq_mistakes = len([row for row in contents if round(row['freq_sum'], 15) != 1.0])
    if num_freq_mistakes > 0:
        logging.warning('%d of selected rows had frequency sum != 1.0 in pileup file %s'
                        % (num_freq_mistakes, pileup_filename))

    # Fill empty locations with placeholders
    covered_indices = [d['location'] for d in contents]
    first_loc, last_loc = contents[0]['location'], contents[-1]['location']
    uncovered_locations = list(set(range(first_loc, last_loc)) - set(covered_indices))
    for uncovered_loc in uncovered_locations:
        contents.append({'location': uncovered_loc,
                         'reference': None, 'nucleotides': None, 'tidy_bases': None,
                         'A_freq': None, 'T_freq': None, 'C_freq': None, 'G_freq': None,
                         'freq_sum': None, 'depth': None, 'qualities': None, 'disregard': None})

    if len(contents) == 0:
        logging.error('No nuclotide locations match criteria in %s.' % pileup_filename)

    return contents

def pileup_df_for_correlation(pileup_ds: List[dict], locations_to_include: List[int]) -> (pd.DataFrame, pd.Series):
    """ Returns dataframe with frequencies from pileup file.

    :param pileup_filename: string filename.
    :return: tuple of dataframe of locations, dataframe of empty locations, average depth, and number of positions
    covered.
    """
    def filter_locations(pileup_row: dict) -> bool:
        return pileup_row['location'] in locations_to_include

    pileup_ds = list(filter(filter_locations, pileup_ds))
    df = pd.DataFrame(pileup_ds)
    df = df.dropna()
    if len(df) == 0:
        logging.error('Dataframe for correlation is empty')

    return df

def add_prediction_to_csv(*args, **kwargs):
    """ Adds prediction to predicted_ages.csv in same folder as this script.

    :param identifier: string identifier for mpileup file
    :param reference_genome: string name of reference genome for mpileup file.
    :return: None

    identifier: str, reference_genome: str,
                          average_pairwise_distance: float, predicted_age: float,
                          avg_depth, num_positions_covered,

    """

    if kwargs:
        assert not args, 'Do not give both key-worded values and values without keywords to be saved.'
        cols = sorted(kwargs.keys())
        values = [str(kwargs[key]) for key in cols]
        if not Path(RESULTS_FILE).is_file():
            logging.debug('Did not find %s, will create with columns %s.' % (RESULTS_FILE, ' '.join(cols)))
            with open(RESULTS_FILE, mode='w') as f:
                f.write('\t'.join(cols) + '\n')
    elif args:
        values = [str(arg) for arg in args]
    else:
        raise ValueError('Either args or kwargs must be provided to add_prediction_to_csv.')

    logging.info('Writing line to csv: %s' % ','.join(values))
    with open(RESULTS_FILE, mode='a') as f:
        observation = '\t'.join(values)
        f.write(observation + '\n')

    return None
