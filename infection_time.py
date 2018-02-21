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

# pol-gene region
POL_START = 2253
POL_END = 5096
POL_LOCATIONS = [loc for loc in list(range(POL_START, POL_END))]

# Model parameters, see Cuevas et al 2015 "Extremely High Mutation Rate of HIV-1 In Vivo."
# Use None to have the program estimate the rates from the data.
MUTATION_RATE = None # 9.3 * 10**(-5)

MIN_DEPTH = 1000

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

        try:
            results_dict = estimate_patient_tsi(contig)
        except Exception as e:
            logging.error('File %s gave error when calculating model estimated time: %s.' % (mpileup, e.args))
            results_dict = {'model_failure': e.args}

        add_prediction_to_csv(identifier=identifier, reference=reference_genome,
                              avg_hamming=average_pairwise_distance, age=predicted_age,
                              **results_dict)
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

def estimate_generations(pg: float, qr: float, qf: float,
                         p0: float = 1.0) -> float:
    """ Estimates number of generations from zero to mutation_rate_g

    Assumes positions are third in either TAC or TAT codons, and should be inside crucial protein-
    coding gene (not nearer than 5% to the end of the gene).

    based on
    g = -(1 / (qr + qf)) * ln[(pg / p0) - qr / (p0*(qr + qf))],
    where
    :param pgs: mutation percentage at generation g of major base (T or C)
    :param qr: rate of "reverse" mutations from minor to major base (T->C for TAC and C->T for TAT)
    :param qf: rate of "forward" mutations from major to minor base (C->T for TAC and T->C for TAT)
    :param p0: initial ratio of major to minor base (C / T for TAC, choose initial sequence for p0=1.0)
    :return: float estimated generation number between original virus and current.
    :raises ValueError: if pg is less than equilibrium level.
    """
    logging.debug('Estimating generations with parameters qr: %.5f, qf: %.5f, p0: %.3f, pg: %.3f'
                  % (qr, qf, p0, pg))
    if qr == 0.0 or qf == 0.0 or p0 == 0.0:
        logging.warning('qr (%.3f), qf (%.3f) or p0 (%.3f) given as 0. Will estimate nan generations.' % (qr, qf, p0))
        return np.nan

    log_arg = (pg / p0) - qr / (p0 * (qr + qf))

    if log_arg < 0.0:
        p_equilibrium = qr / (qr + qf)
        raise ValueError('Estimating generation number with mutation frequency less '
                         'than equilibrium is not possible.\n Mutation frequency given '
                         'is pg = %.2f, whereas equilibrium is %.2f. Other values are '
                         'qr (%.5f), qf (%.5f) and p0 (%.3f).' % (pg, p_equilibrium, qr, qf, p0))

    return -(1 / (qr + qf)) * np.log(log_arg)

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
    Codon = namedtuple('Codon', ['codon_string', 'first', 'second', 'third', 'codon_pos'])
    codons = []
    for first_loc in range(0, len(contig), 3):
        codon_string = s0_sequence(contig[first_loc: min(first_loc + 3, len(contig))])
        first = contig[first_loc]
        second = contig[first_loc + 1] if first_loc + 1 < len(contig) else dict
        third = contig[first_loc + 2] if first_loc + 2 < len(contig) else dict
        codons.append(Codon(codon_string, first, second, third, first_loc))

    return codons

def estimate_qct(codons: List) -> (float, List[int], int):
    num_five_percent_codons = len(codons) // 20
    sdcc_codons = [codon for codon in codons[:-num_five_percent_codons]
                   if (codon.codon_string in ['CAG', 'CAA', 'CGA']
                       and len(codon.first['tidy_bases']) > MIN_DEPTH)]

    cag_codons = [codon for codon in sdcc_codons if codon.codon_string == 'CAG']
    caa_codons = [codon for codon in sdcc_codons if codon.codon_string == 'CAA']
    cga_codons = [codon for codon in sdcc_codons if codon.codon_string == 'CGA']

    if len(cga_codons + caa_codons + cga_codons) < 1:
        raise ValueError('Could not find any SDCC codons')

    cag_depths = [str(len(codon.first['tidy_bases'])) for codon in cag_codons]
    caa_depths = [str(len(codon.first['tidy_bases'])) for codon in caa_codons]
    cga_depths = [str(len(codon.first['tidy_bases'])) for codon in cga_codons]

    outliers = [codon.first['T_freq'] for codon in caa_codons if codon.first['T_freq'] > 0.01]
    print('There are %d outliers with mean %.3f' % (len(outliers), np.mean(outliers)))


    cag_estimates = [codon.first['T_freq'] for codon in cag_codons]
    caa_estimates = [codon.first['T_freq'] for codon in caa_codons if codon.first['T_freq'] < 0.01]
    cga_estimates = [codon.first['T_freq'] for codon in cga_codons]

    cag_qct_mean = sum(cag_estimates) / len(cag_estimates) if len(cag_estimates) > 0 else 0
    caa_qct_mean = sum(caa_estimates) / len(caa_estimates) if len(caa_estimates) > 0 else 0
    cga_qct_mean = sum(cga_estimates) / len(cga_estimates) if len(cga_estimates) > 0 else 0

    results = {'death_freq_cag': cag_qct_mean, 'cag_depths': '|'.join(cag_depths), 'num_cag': len(cag_estimates),
               'death_freq_caa': caa_qct_mean, 'caa_depths': '|'.join(caa_depths), 'num_caa': len(caa_estimates),
               'death_freq_cga': cga_qct_mean, 'cga_depths': '|'.join(cga_depths), 'num_cga': len(cga_estimates)}

    qct_mean = (len(cag_estimates)*cag_qct_mean
                + len(caa_estimates)*caa_qct_mean
                + len(cga_estimates)*cga_qct_mean) / (len(cag_estimates) + len(caa_estimates) + len(cga_estimates))

    return qct_mean, results

def estimate_qtc(codons_contig: List[str], qct) -> float:
    assert type(qct) is float, 'estimate_qtc requires float value for qct (given %s).' % type(qct)
    num_tac = codons_contig.count('TAC')
    num_tat = codons_contig.count('TAT')
    r = num_tac / num_tat
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
    qct_mean, qct_descriptives = estimate_qct(codons) if MUTATION_RATE is None else (MUTATION_RATE, {})
    codon_strings = [c.codon_string for c in codons]
    qtc_mean = MUTATION_RATE or estimate_qtc(codon_strings, qct=qct_mean)

    p0 = 1.0
    tac_codons = [codon for codon in codons
                  if codon.codon_string == 'TAC' and len(codon.third['tidy_bases']) > MIN_DEPTH]
    tac_pgs = [codon.third['C_freq'] / (codon.third['C_freq'] + codon.third['T_freq'])
               for codon in tac_codons]
    tat_codons = [codon for codon in codons
                  if codon.codon_string == 'TAT' and len(codon.third['tidy_bases']) > MIN_DEPTH]
    tat_pgs = [codon.third['T_freq'] / (codon.third['C_freq'] + codon.third['T_freq'])
               for codon in tat_codons]

    str_g_depths = [str(len(codon.third['tidy_bases'])) for codon in tac_codons + tat_codons]

    num_tac = len(tac_codons)
    num_tat = len(tat_codons)

    tac_pg = sum(tac_pgs) / num_tac
    tat_pg = sum(tat_pgs) / num_tat

    try:
        tac_g_mean = round(estimate_generations(tac_pg, qtc_mean, qct_mean, p0), 1)
        tat_g_mean = round(estimate_generations(tat_pg, qct_mean, qtc_mean, p0), 1)
        g_mean = (tac_g_mean*num_tac + tat_g_mean*num_tat) / (num_tat + num_tac)
    except Exception as e:
        logging.error('Could not estimate generations: %s' % e.args)
        raise e

    results = {'g_mean': g_mean, 'g_tac': tac_g_mean, 'g_tat': tat_g_mean,
               'num_tat': num_tat, 'num_tac': num_tac, 'model_depths': '|'.join(str_g_depths),
               'p0': p0, 'tat_pg': tat_pg, 'tac_pg': tac_pg,
               'q_tc': qtc_mean, 'q_ct': qct_mean,
               'model_years': round((g_mean * days_per_generation) / 365, 3)}
    results.update(qct_descriptives)

    return results

if __name__ == '__main__':
    main()
