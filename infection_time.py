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
from typing import List
from tsi_filehandling import (get_relevant_mpileup_files, parse_pileup_to_contig, pileup_df_for_correlation,
                              add_prediction_to_csv, get_relevant_mpileup_files, read_identifier, read_reference_genome)


# Puller et al. method parameters
SLOPE = 250.28
INTERCEPT = -0.08
THRESHOLD = 0.0
MIN_DEPTH = 200

# pol-gene region
POL_START = 2253
POL_END = 5096
POL_LOCATIONS = [loc for loc in list(range(POL_START, POL_END))]

# Model parameters, see Cuevas et al 2015 "Extremely High Mutation Rate of HIV-1 In Vivo."
MUTATION_RATE = 9.3 * 10**(-5)

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
        contig = parse_pileup_to_contig(mpileup, start=POL_START, end=POL_END)

        locations_for_corr_method = POL_LOCATIONS[2::3]
        pileup_for_correlation = pileup_df_for_correlation(contig, locations_for_corr_method)
        pileup_for_correlation = pileup_for_correlation[pileup_for_correlation['tidy_bases'].str.len() > MIN_DEPTH]
        num_locs_for_correlation = len(pileup_for_correlation)

        try:
            (average_pairwise_distance,
             predicted_age) = predict_age(pileup_for_correlation)
        except Exception as e:
            logging.error('File %s gave error when calculating average pairwise distance: %s.' % (mpileup, e.args))
            average_pairwise_distance = np.nan
            predicted_age = np.nan

        tsi_g, tsi_years, p0, pg, qtc_mean, qct_mean, num_loc_model = estimate_patient_tsi(contig)

        add_prediction_to_csv(identifier=identifier, reference=reference_genome,
                              avg_hamming=average_pairwise_distance, age=predicted_age,
                              g_model=tsi_g, years_model=tsi_years, p0=p0, pg=pg, qtc=qtc_mean,
                              qct=qct_mean, num_loc_corr=num_locs_for_correlation, corr_min_depth=MIN_DEPTH,
                              num_loc_model=num_loc_model)
        logging.info('Predicted age %s from Hamming distance %.3f for identifier %s (reference %s).'
                     % (predicted_age, average_pairwise_distance, identifier, reference_genome))

    logging.info('Completed all calculations, now finishing in good form.')
    return 0

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

    average_pairwise_distance = sum(parsed_pileup_df['hamming']) / len(parsed_pileup_df)

    return average_pairwise_distance

def find_potential_stop_codons(protein_coding_nucleotides: str, offset = 0) -> List[dict]:
    """ Returns list of dicts with {position: nucleotide which creates stop codon}.

    Important: This function removes the last 5% of the coding region provided, so don't do this yourself first.

    :param protein_coding_nucleotides: Sequence of nucleotides, the complete coding sequence for a given protein.
    :return: list of dictionaries {position: nucleotide which creates stop codon}
    """
    logging.debug('Starting to determine potential stop-codon locations.')
    pns = protein_coding_nucleotides
    assert len(pns) % 3 == 0, 'open reading frame must have length divisible by 3.'
    num_codons = len(pns) // 3
    logging.debug('There are %d codons.' % num_codons)
    num_codons_to_cut = num_codons // 20
    logging.debug('We will cut %d codons.' % num_codons_to_cut)
    num_nucleotides_to_cut = num_codons_to_cut * 3
    logging.debug('We will cut %d nucleotides.' % num_nucleotides_to_cut)
    pns = pns[:-num_nucleotides_to_cut] if num_codons_to_cut > 0 else pns# removes last 5% of codons
    assert len(pns) % 3 == 0, 'Error in code created open reading frame with length not divisible by 3.'
    logging.debug('Having cut off the end, we now use:\n%s' % pns)

    pstop_codons = {'TGC': [('A', 3)], 'AAG': [('T', 1)], 'TTG': [('A', 2)], 'TCA': [('G', 2), ('A', 2)],
                    'TAT': [('G', 3), ('A', 3)], 'TAC': [('G', 3), ('A', 3)], 'TTA': [('A', 2), ('G', 2)],
                    'TGG': [('A', 2), ('A', 3)], 'GGA': [('T', 1)], 'CAG': [('T', 1)], 'CAA': [('T', 1)],
                    'AAA': [('T', 1)], 'TCG': [('A', 2)], 'AGA': [('T', 1)], 'GAA': [('T', 1)], 'GAG': [('T', 1)],
                    'CGA': [('T', 1)], 'TGT': [('A', 3)]}

    pstop_positions = []
    for codon_num in range(0, len(pns) // 3):
        codon = pns[codon_num*3:codon_num*3 + 3]
        for match in pstop_codons.get(codon, []):
            pstop_positions.append((match[0], codon_num*3 + match[1] + offset))

    return pstop_positions

def estimate_generations(pg: float, qtc: float, qct: float,
                         p0: float = None) -> float:
    """ Estimates number of generations from zero to mutation_rate_g

    Assumes positions are third in TAC or TAT codons, and should be inside crucial protein-
    coding gene (not nearer than 5% to the end of the gene).

    based on
    g = -(1 / (qi + qo)) * ln[(pg / p0) - qi / (p0*(qi + qo))],
    where
    :param pgs: mutation percentage at generation g of major base
    :param qtc: rate of mutations from T to C
    :param qct: rate of mutations from C to T
    :param p0: initial ratio of C to T
    :return: float estimated generation number between original virus and current.
    :raises ValueError: if pg is less than equilibrium level.
    """
    logging.debug('Estimating generations with parameters qtc: %.5f, qct: %.5f, p0: %.3f, pg: %.3f'
                  % (qtc, qct, p0, pg))
    if qtc == 0.0 or qct == 0.0 or p0 == 0.0:
        logging.warning('qtc (%.3f), qct (%.3f) or p0 (%.3f) given as 0. Will estimate nan generations.' % (qtc, qct, p0))
        return np.nan

    p_equilibrium = qtc / (qtc + qct)
    log_arg = (pg / p0) - qtc / (p0 * (qtc + qct))

    if log_arg < 0.0:
        raise ValueError('Estimating generation number with mutation frequency less '
                         'than equilibrium is not possible.\n Mutation frequency given '
                         'is pg = %.2f, whereas equilibrium is %.2f. Other values are '
                         'qtc (%.5f), qct (%.5f) and p0 (%.3f).' % (pg, p_equilibrium, qtc, qct, p0))

    return -(1 / (qtc + qct)) * np.log(log_arg)

def get_S0_base(pileup: dict) -> str:
    if pileup['freq_sum'] is None:
        return ''

    freqs_dict = lambda d: {key: value for key, value in d.items() if '_freq' in key}
    major_base = lambda freqs_dict: max(freqs_dict.items(), key=lambda d: d[1])[0][0]
    get_S0_base = lambda row_dict: major_base(freqs_dict(row_dict))
    return get_S0_base(pileup)

def s0_sequence(pileup: List[dict]):
    """ Returns the sequence of nucleotides based on major threshold.
    """
    s0 = ''.join([get_S0_base(d) for d in pileup])
    return s0

def codonify(contig: List[dict], s0=None):
    from collections import namedtuple
    Codon = namedtuple('Codon', ['codon_string', 'first', 'second', 'third'])
    codons = []
    for first_loc in range(0, len(contig), 3):
        codon_string = s0_sequence(contig[first_loc: min(first_loc + 3, len(contig))])
        first = contig[first_loc]
        second = contig[first_loc + 1] if first_loc + 1 < len(contig) else dict
        third = contig[first_loc + 2] if first_loc + 2 < len(contig) else dict
        codons.append(Codon(codon_string, first, second, third))

    return codons

def estimate_qct(codons: List) -> float:
    sdcc_codons = [codon for codon in codons if codon.codon_string in ['CAG', 'CAA', 'CGA']]
    num_five_percent_codons = len(sdcc_codons) // 20
    sdcc_codons = sdcc_codons[:-num_five_percent_codons]

    estimates = [codon.first.get('T_freq', None) for codon in sdcc_codons]
    #estimates = [estimate for estimate in estimates if estimate is not None and estimate < 0.01]
    #co_e = list(zip([codon.codon_string for codon in sdcc_codons], estimates))
    #print(co_e)
    qct_mean = sum(estimates) / len(estimates)
    if qct_mean > 0.0005:
        logging.warning('Weirdly large q_CT estimated from S_DCC (%.3f).' % qct_mean)
    return qct_mean

def estimate_qtc(codons_contig: List[str], qct) -> (float, float, float):
    r = codons_contig.count('TAC') / codons_contig.count('TAT')
    qtc = r * qct
    return qtc

def estimate_patient_tsi(contig: List[dict], days_per_generation=1.2) -> (float, float):
    """ Returns tuple of mean, min and max estimates of time since infection (generations).

    The pileup list of dictionaries must include keys A_freq, T_freq etc, and must cover
    a continuous protein-coding region.

    :param: pileup, list of dictionaries with keys A_freq, T_freq etc. for nucleotide frequencies.
    :return: 3-tuple with mean, min and max estimates of time since infections (in generations)
    """
    s0 = s0_sequence(contig)
    start_codon_loc = s0.find('CTC')
    if start_codon_loc > 0:
        contig = contig[start_codon_loc:]
        logging.debug('Start-codon CTC found at location %d in sequence:\n%s' % (start_codon_loc, s0))
        s0 = s0[start_codon_loc:]
    elif start_codon_loc == 0:
        logging.debug('Found start-codon CTC at expected first location.')
    else:
        logging.error('Could not find start codon in given sequence:\n%s' % s0)

    codons = codonify(contig)
    qct_mean = MUTATION_RATE or estimate_qct(codons)
    codon_strings = [c.codon_string for c in codons]
    qtc_mean = MUTATION_RATE or estimate_qtc(codon_strings, qct=qct_mean)

    p0 = 1.0
    tac_codons = [codon for codon in codons
                  if codon.codon_string == 'TAC' and len(codon.third['tidy_bases']) > MIN_DEPTH]
    pgs = [codon.third['C_freq'] / (codon.third['C_freq'] + codon.third['T_freq'])
           for codon in tac_codons]
    pg = sum(pgs) / len(pgs)

    try:
        g_mean = estimate_generations(pg, qtc_mean, qct_mean, p0)
    except Exception as e:
        print('p0: ', p0)
        print('qct: ', qct_mean)
        print('qtc: ', qtc_mean)
        print('pg: ', pg)
        raise e

    return g_mean, (g_mean * days_per_generation) / 365, p0, pg, qtc_mean, qct_mean, len(tac_codons)

if __name__ == '__main__':
    main()
