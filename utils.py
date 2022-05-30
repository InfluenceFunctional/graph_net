"""import statement"""
import numpy as np
import numba as nb
import os
import pandas as pd
from argparse import Namespace
import yaml
from pathlib import Path

'''
general utilities
'''


def initialize_metrics_dict(metrics):
    m_dict = {}
    for metric in metrics:
        m_dict[metric] = []

    return m_dict


def printRecord(statement):
    """
    print a string to command line output and a text file
    :param statement:
    :return:
    """
    print(statement)
    if os.path.exists('record.txt'):
        with open('record.txt', 'a') as file:
            file.write('\n' + statement)
    else:
        with open('record.txt', 'w') as file:
            file.write('\n' + statement)

def add_bool_arg(parser, name, default=False):
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--' + name, dest=name, action='store_true')
    group.add_argument('--no-' + name, dest=name, action='store_false')
    parser.set_defaults(**{name: default})


def add_arg_list(parser, arg_list):
    for entry in arg_list:
        if entry['type'] == 'bool':
            add_bool_arg(parser, entry['name'], entry['default'])
        else:
            parser.add_argument('--' + entry['name'], type=entry['type'], default=entry['default'])

    return parser



class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def get_n_config(model):
    """
    count parameters for a pytorch model
    :param model:
    :return:
    """
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


def chunkify(lst, n):
    return [lst[i::n] for i in range(n)]


def standardize(data, return_std=False, known_mean=None, known_std=None):
    data = data.astype('float32')
    if known_mean is not None:
        mean = known_mean
    else:
        mean = np.mean(data)

    if known_std is not None:
        std = known_std
    else:
        std = np.std(data)

    std_data = (data - mean) / std

    if return_std:
        return std_data, mean, std
    else:
        return std_data


def loadSpaceGroups():
    sgDict = pd.DataFrame(columns=['system', 'laue', 'class', 'ce', 'SG'], index=np.arange(0, 230 + 1))
    sgDict.loc[0] = pd.Series({'system': 'n/a', 'laue': 'n/a', 'class': 'n/a', 'ce': 'n/a', 'SG': 'n/a'})  # add an entry for misc / throwaway categories

    # Triclinic
    i = 1
    sgDict.loc[i] = pd.Series({'system': 'triclinic',
                               'laue': '-1',
                               'class': '1',
                               'ce': 'p',
                               'SG': 'P1'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'triclinic',
                               'laue': '-1',
                               'class': '-1',
                               'ce': 'p',
                               'SG': 'P-1'
                               })
    i += 1

    # Monoclinic
    sgDict.loc[i] = pd.Series({'system': 'monoclinic',
                               'laue': '2/m',
                               'class': '2',
                               'ce': 'p',
                               'SG': 'P2'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'monoclinic',
                               'laue': '2/m',
                               'class': '2',
                               'ce': 'p',
                               'SG': 'P21'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'monoclinic',
                               'laue': '2/m',
                               'class': '2',
                               'ce': 'c',
                               'SG': 'C2'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'monoclinic',
                               'laue': '2/m',
                               'class': 'm',
                               'ce': 'p',
                               'SG': 'Pm'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'monoclinic',
                               'laue': '2/m',
                               'class': 'm',
                               'ce': 'p',
                               'SG': 'Pc'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'monoclinic',
                               'laue': '2/m',
                               'class': 'm',
                               'ce': 'c',
                               'SG': 'Cm'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'monoclinic',
                               'laue': '2/m',
                               'class': 'm',
                               'ce': 'c',
                               'SG': 'Cc'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'monoclinic',
                               'laue': '2/m',
                               'class': '2/m',
                               'ce': 'p',
                               'SG': 'P2/m'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'monoclinic',
                               'laue': '2/m',
                               'class': '2/m',
                               'ce': 'p',
                               'SG': 'P21/m'  # alt - P21m
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'monoclinic',
                               'laue': '2/m',
                               'class': '2/m',
                               'ce': 'c',
                               'SG': 'C2/m'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'monoclinic',
                               'laue': '2/m',
                               'class': '2/m',
                               'ce': 'p',
                               'SG': 'P2/c'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'monoclinic',
                               'laue': '2/m',
                               'class': '2/m',
                               'ce': 'p',
                               'SG': 'P21/c'  # alt - P21c
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'monoclinic',
                               'laue': '2/m',
                               'class': '2/m',
                               'ce': 'c',
                               'SG': 'C2/c'
                               })
    i += 1

    # Orthorhombic
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': '222',
                               'ce': 'p',
                               'SG': 'P222'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': '222',
                               'ce': 'p',
                               'SG': 'P2221'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': '222',
                               'ce': 'p',
                               'SG': 'P21212'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': '222',
                               'ce': 'p',
                               'SG': 'P212121'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': '222',
                               'ce': 'c',
                               'SG': 'C2221'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': '222',
                               'ce': 'c',
                               'SG': 'C222'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': '222',
                               'ce': 'f',
                               'SG': 'F222'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': '222',
                               'ce': 'i',
                               'SG': 'I222'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': '222',
                               'ce': 'i',
                               'SG': 'I212121'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mm2',
                               'ce': 'p',
                               'SG': 'Pmm2'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mm2',
                               'ce': 'p',
                               'SG': 'Pmc21'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mm2',
                               'ce': 'p',
                               'SG': 'Pcc2'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mm2',
                               'ce': 'p',
                               'SG': 'Pma2'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mm2',
                               'ce': 'p',
                               'SG': 'Pca21'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mm2',
                               'ce': 'p',
                               'SG': 'Pnc2'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mm2',
                               'ce': 'p',
                               'SG': 'Pmn21'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mm2',
                               'ce': 'p',
                               'SG': 'Pba2'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mm2',
                               'ce': 'p',
                               'SG': 'Pna21'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mm2',
                               'ce': 'p',
                               'SG': 'Pnn2'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mm2',
                               'ce': 'c',  # or a
                               'SG': 'Cmm2'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mm2',
                               'ce': 'c',  # or a
                               'SG': 'Cmc21'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mm2',
                               'ce': 'c',  # or a
                               'SG': 'Ccc2'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mm2',
                               'ce': 'c',  # or a
                               'SG': 'Amm2'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mm2',
                               'ce': 'c',  # or a
                               'SG': 'Abm2'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mm2',
                               'ce': 'c',  # or a
                               'SG': 'Ama2'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mm2',
                               'ce': 'c',  # or a
                               'SG': 'Aba2'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mm2',
                               'ce': 'f',
                               'SG': 'Fmm2'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mm2',
                               'ce': 'f',
                               'SG': 'Fdd2'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mm2',
                               'ce': 'i',
                               'SG': 'Imm2'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mm2',
                               'ce': 'i',
                               'SG': 'Iba2'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mm2',
                               'ce': 'i',
                               'SG': 'Ima2'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mmm',
                               'ce': 'p',
                               'SG': 'Pmmm'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mmm',
                               'ce': 'p',
                               'SG': 'Pnnn'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mmm',
                               'ce': 'p',
                               'SG': 'Pccm'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mmm',
                               'ce': 'p',
                               'SG': 'Pban'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mmm',
                               'ce': 'p',
                               'SG': 'Pmma'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mmm',
                               'ce': 'p',
                               'SG': 'Pnna'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mmm',
                               'ce': 'p',
                               'SG': 'Pmna'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mmm',
                               'ce': 'p',
                               'SG': 'Pcca'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mmm',
                               'ce': 'p',
                               'SG': 'Pbam'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mmm',
                               'ce': 'p',
                               'SG': 'Pccn'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mmm',
                               'ce': 'p',
                               'SG': 'Pbcm'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mmm',
                               'ce': 'p',
                               'SG': 'Pnnm'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mmm',
                               'ce': 'p',
                               'SG': 'Pmmn'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mmm',
                               'ce': 'p',
                               'SG': 'Pbcn'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mmm',
                               'ce': 'p',
                               'SG': 'Pbca'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mmm',
                               'ce': 'p',
                               'SG': 'Pnma'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mmm',
                               'ce': 'c',
                               'SG': 'Cmcm'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mmm',
                               'ce': 'c',
                               'SG': 'Cmca'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mmm',
                               'ce': 'c',
                               'SG': 'Cmmm'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mmm',
                               'ce': 'c',
                               'SG': 'Cccm'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mmm',
                               'ce': 'c',
                               'SG': 'Cmma'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mmm',
                               'ce': 'c',
                               'SG': 'Ccca'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mmm',
                               'ce': 'f',
                               'SG': 'Fmmm'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mmm',
                               'ce': 'f',
                               'SG': 'Fddd'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mmm',
                               'ce': 'i',
                               'SG': 'Immm'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mmm',
                               'ce': 'i',
                               'SG': 'Ibam'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mmm',
                               'ce': 'i',
                               'SG': 'Ibca'  # aka Ibcm
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'orthorhombic',
                               'laue': 'mmm',
                               'class': 'mmm',
                               'ce': 'i',
                               'SG': 'Imma'
                               })
    i += 1

    # Tetragonal
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/m',
                               'class': '4',
                               'ce': 'p',
                               'SG': 'P4'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/m',
                               'class': '4',
                               'ce': 'p',
                               'SG': 'P41'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/m',
                               'class': '4',
                               'ce': 'p',
                               'SG': 'P42'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/m',
                               'class': '4',
                               'ce': 'p',
                               'SG': 'P43'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/m',
                               'class': '4',
                               'ce': 'i',
                               'SG': 'I4'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/m',
                               'class': '4',
                               'ce': 'i',
                               'SG': 'I41'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/m',
                               'class': '-4',
                               'ce': 'p',
                               'SG': 'P-4'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/m',
                               'class': '-4',
                               'ce': 'i',
                               'SG': 'I-4'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/m',
                               'class': '4/m',
                               'ce': 'p',
                               'SG': 'P4/m'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/m',
                               'class': '4/m',
                               'ce': 'p',
                               'SG': 'P42/m'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/m',
                               'class': '4/m',
                               'ce': 'p',
                               'SG': 'P4/n'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/m',
                               'class': '4/m',
                               'ce': 'p',
                               'SG': 'P42/n'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/m',
                               'class': '4/m',
                               'ce': 'i',
                               'SG': 'I4/m'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/m',
                               'class': '4/m',
                               'ce': 'i',
                               'SG': 'I41/a'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '422',
                               'ce': 'p',
                               'SG': 'P422'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '422',
                               'ce': 'p',
                               'SG': 'P4212'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '422',
                               'ce': 'p',
                               'SG': 'P4122'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '422',
                               'ce': 'p',
                               'SG': 'P41212'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '422',
                               'ce': 'p',
                               'SG': 'P4222'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '422',
                               'ce': 'p',
                               'SG': 'P42212'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '422',
                               'ce': 'p',
                               'SG': 'P4322'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '422',
                               'ce': 'p',
                               'SG': 'P43212'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '422',
                               'ce': 'i',
                               'SG': 'I422'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '422',
                               'ce': 'i',
                               'SG': 'I4122'  # aka I4212
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '4mm',
                               'ce': 'p',
                               'SG': 'P4mm'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '4mm',
                               'ce': 'p',
                               'SG': 'P4bm'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '4mm',
                               'ce': 'p',
                               'SG': 'P42cm'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '4mm',
                               'ce': 'p',
                               'SG': 'P42nm'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '4mm',
                               'ce': 'p',
                               'SG': 'P4cc'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '4mm',
                               'ce': 'p',
                               'SG': 'P4nc'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '4mm',
                               'ce': 'p',
                               'SG': 'P42mc'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '4mm',
                               'ce': 'p',
                               'SG': 'P42bc'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '4mm',
                               'ce': 'i',
                               'SG': 'I4mm'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '4mm',
                               'ce': 'i',
                               'SG': 'I4cm'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '4mm',
                               'ce': 'i',
                               'SG': 'I41md'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '4mm',
                               'ce': 'i',
                               'SG': 'I41cd'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '-42m',
                               'ce': 'p',
                               'SG': 'P-42m'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '-42m',
                               'ce': 'p',
                               'SG': 'P-42c'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '-42m',
                               'ce': 'p',
                               'SG': 'P-421m'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '-42m',
                               'ce': 'p',
                               'SG': 'P-421c'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '-4m2',
                               'ce': 'p',
                               'SG': 'P-4m2'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '-4m2',
                               'ce': 'p',
                               'SG': 'P-4c2'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '-4m2',
                               'ce': 'p',
                               'SG': 'P-4b2'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '-4m2',
                               'ce': 'p',
                               'SG': 'P-4n2'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '-4m2',
                               'ce': 'i',
                               'SG': 'I-4m2'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '-4m2',
                               'ce': 'i',
                               'SG': 'I-4c2'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '-42m',
                               'ce': 'i',
                               'SG': 'I-42m'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '-42m',
                               'ce': 'p',
                               'SG': 'I-42d'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '4/mmm',
                               'ce': 'p',
                               'SG': 'P4/mmm'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '4/mmm',
                               'ce': 'p',
                               'SG': 'P4/mcc'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '4/mmm',
                               'ce': 'p',
                               'SG': 'P4/nbm'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '4/mmm',
                               'ce': 'p',
                               'SG': 'P4/nnc'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '4/mmm',
                               'ce': 'p',
                               'SG': 'P4/mbm'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '4/mmm',
                               'ce': 'p',
                               'SG': 'P4/mnc'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '4/mmm',
                               'ce': 'p',
                               'SG': 'P4/nmm'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '4/mmm',
                               'ce': 'p',
                               'SG': 'P4/ncc'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '4/mmm',
                               'ce': 'p',
                               'SG': 'P42/mmc'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '4/mmm',
                               'ce': 'p',
                               'SG': 'P42/mcm'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '4/mmm',
                               'ce': 'p',
                               'SG': 'P42/nbc'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '4/mmm',
                               'ce': 'p',
                               'SG': 'P42/nnm'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '4/mmm',
                               'ce': 'p',
                               'SG': 'P42/mbc'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '4/mmm',
                               'ce': 'p',
                               'SG': 'P42/mnm'  # incorrectly /mcm in source
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '4/mmm',
                               'ce': 'p',
                               'SG': 'P42/nmc'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '4/mmm',
                               'ce': 'p',
                               'SG': 'P42/ncm'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '4/mmm',
                               'ce': 'i',
                               'SG': 'I4/mmm'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '4/mmm',
                               'ce': 'i',
                               'SG': 'I4/mcm'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '4/mmm',
                               'ce': 'i',
                               'SG': 'I41/amd'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'tetragonal',
                               'laue': '4/mmm',
                               'class': '4/mmm',
                               'ce': 'i',
                               'SG': 'I41/acd'
                               })
    i += 1

    # Trigonal
    sgDict.loc[i] = pd.Series({'system': 'trigonal',
                               'laue': '-3',
                               'class': '3',
                               'ce': 'p',
                               'SG': 'P3'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'trigonal',
                               'laue': '-3',
                               'class': '3',
                               'ce': 'p',
                               'SG': 'P31'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'trigonal',
                               'laue': '-3',
                               'class': '3',
                               'ce': 'p',
                               'SG': 'P32'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'trigonal',
                               'laue': '-3',
                               'class': '3',
                               'ce': 'r',
                               'SG': 'R3'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'trigonal',
                               'laue': '-3',
                               'class': '-3',
                               'ce': 'p',
                               'SG': 'P-3'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'trigonal',
                               'laue': '-3',
                               'class': '-3',
                               'ce': 'r',
                               'SG': 'R-3'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'trigonal',
                               'laue': '-3m',
                               'class': '312',
                               'ce': 'p',
                               'SG': 'P312'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'trigonal',
                               'laue': '-3m',
                               'class': '321',
                               'ce': 'p',
                               'SG': 'P321'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'trigonal',
                               'laue': '-3m',
                               'class': '312',
                               'ce': 'p',
                               'SG': 'P3112'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'trigonal',
                               'laue': '-3m',
                               'class': '321',
                               'ce': 'p',
                               'SG': 'P3121'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'trigonal',
                               'laue': '-3m',
                               'class': '312',
                               'ce': 'p',
                               'SG': 'P3212'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'trigonal',
                               'laue': '-3m',
                               'class': '321',
                               'ce': 'p',
                               'SG': 'P3221'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'trigonal',
                               'laue': '-3m',
                               'class': '321',
                               'ce': 'r',
                               'SG': 'R32'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'trigonal',
                               'laue': '-3m',
                               'class': '3m1',
                               'ce': 'p',
                               'SG': 'P3m1'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'trigonal',
                               'laue': '-3m',
                               'class': '31m',
                               'ce': 'p',
                               'SG': 'P31m'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'trigonal',
                               'laue': '-3m',
                               'class': '3m1',
                               'ce': 'p',
                               'SG': 'P3c1'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'trigonal',
                               'laue': '-3m',
                               'class': '31m',
                               'ce': 'p',
                               'SG': 'P31c'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'trigonal',
                               'laue': '-3m',
                               'class': '3m1',
                               'ce': 'r',
                               'SG': 'R3m'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'trigonal',
                               'laue': '-3m',
                               'class': '3m1',
                               'ce': 'r',
                               'SG': 'R3c'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'trigonal',
                               'laue': '-3m',
                               'class': '-31m',
                               'ce': 'p',
                               'SG': 'P-31m'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'trigonal',
                               'laue': '-3m',
                               'class': '-31m',
                               'ce': 'p',
                               'SG': 'P-31c'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'trigonal',
                               'laue': '-3m',
                               'class': '-3m1',
                               'ce': 'p',
                               'SG': 'P-3m1'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'trigonal',
                               'laue': '-3m',
                               'class': '-3m1',
                               'ce': 'p',
                               'SG': 'P-3c1'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'trigonal',
                               'laue': '-3m',
                               'class': '-3m1',
                               'ce': 'r',
                               'SG': 'R-3m'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'trigonal',
                               'laue': '-3m',
                               'class': '-3m1',
                               'ce': 'r',
                               'SG': 'R-3c'
                               })
    i += 1

    # Hexagonal
    sgDict.loc[i] = pd.Series({'system': 'hexagonal',
                               'laue': '6/m',
                               'class': '6',
                               'ce': 'p',
                               'SG': 'P6'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'hexagonal',
                               'laue': '6/m',
                               'class': '6',
                               'ce': 'p',
                               'SG': 'P61'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'hexagonal',
                               'laue': '6/m',
                               'class': '6',
                               'ce': 'p',
                               'SG': 'P65'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'hexagonal',
                               'laue': '6/m',
                               'class': '6',
                               'ce': 'p',
                               'SG': 'P62'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'hexagonal',
                               'laue': '6/m',
                               'class': '6',
                               'ce': 'p',
                               'SG': 'P64'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'hexagonal',
                               'laue': '6/m',
                               'class': '6',
                               'ce': 'p',
                               'SG': 'P63'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'hexagonal',
                               'laue': '6/m',
                               'class': '-6',
                               'ce': 'p',
                               'SG': 'P-6'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'hexagonal',
                               'laue': '6/m',
                               'class': '6/m',
                               'ce': 'p',
                               'SG': 'P6/m'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'hexagonal',
                               'laue': '6/m',
                               'class': '6/m',
                               'ce': 'p',
                               'SG': 'P63/m'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'hexagonal',
                               'laue': '6/mmm',
                               'class': '622',
                               'ce': 'p',
                               'SG': 'P622'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'hexagonal',
                               'laue': '6/mmm',
                               'class': '622',
                               'ce': 'p',
                               'SG': 'P6122'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'hexagonal',
                               'laue': '6/mmm',
                               'class': '622',
                               'ce': 'p',
                               'SG': 'P6522'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'hexagonal',
                               'laue': '6/mmm',
                               'class': '622',
                               'ce': 'p',
                               'SG': 'P6222'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'hexagonal',
                               'laue': '6/mmm',
                               'class': '622',
                               'ce': 'p',
                               'SG': 'P6422'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'hexagonal',
                               'laue': '6/mmm',
                               'class': '622',
                               'ce': 'p',
                               'SG': 'P6322'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'hexagonal',
                               'laue': '6/mmm',
                               'class': '6mm',
                               'ce': 'p',
                               'SG': 'P6mm'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'hexagonal',
                               'laue': '6/mmm',
                               'class': '6mm',
                               'ce': 'p',
                               'SG': 'P6cc'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'hexagonal',
                               'laue': '6/mmm',
                               'class': '6mm',
                               'ce': 'p',
                               'SG': 'P63cm'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'hexagonal',
                               'laue': '6/mmm',
                               'class': '6mm',
                               'ce': 'p',
                               'SG': 'P63mc'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'hexagonal',
                               'laue': '6/mmm',
                               'class': '-6m2',
                               'ce': 'p',
                               'SG': 'P-6m2'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'hexagonal',
                               'laue': '6/mmm',
                               'class': '-6m2',
                               'ce': 'p',
                               'SG': 'P-6c2'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'hexagonal',
                               'laue': '6/mmm',
                               'class': '-62m',
                               'ce': 'p',
                               'SG': 'P-62m'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'hexagonal',
                               'laue': '6/mmm',
                               'class': '-62m',
                               'ce': 'p',
                               'SG': 'P-62c'  # error in source, missing negative http://pd.chem.ucl.ac.uk/pdnn/symm3/allsgp.htm
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'hexagonal',
                               'laue': '6/mmm',
                               'class': '6/mmm',
                               'ce': 'p',
                               'SG': 'P6/mmm'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'hexagonal',
                               'laue': '6/mmm',
                               'class': '6/mmm',
                               'ce': 'p',
                               'SG': 'P6/mcc'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'hexagonal',
                               'laue': '6/mmm',
                               'class': '6/mmm',
                               'ce': 'p',
                               'SG': 'P63/mcm'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'hexagonal',
                               'laue': '6/mmm',
                               'class': '6/mmm',
                               'ce': 'p',
                               'SG': 'P63/mmc'
                               })
    i += 1

    # Cubic
    sgDict.loc[i] = pd.Series({'system': 'cubic',
                               'laue': 'm-3',
                               'class': '23',
                               'ce': 'p',
                               'SG': 'P23'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'cubic',
                               'laue': 'm-3',
                               'class': '23',
                               'ce': 'f',
                               'SG': 'F23'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'cubic',
                               'laue': 'm-3',
                               'class': '23',
                               'ce': 'i',
                               'SG': 'I23'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'cubic',
                               'laue': 'm-3',
                               'class': '23',
                               'ce': 'p',
                               'SG': 'P213'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'cubic',
                               'laue': 'm-3',
                               'class': '23',
                               'ce': 'i',
                               'SG': 'I213'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'cubic',
                               'laue': 'm-3',
                               'class': 'm-3',
                               'ce': 'p',
                               'SG': 'Pm-3'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'cubic',
                               'laue': 'm-3',
                               'class': 'm-3',
                               'ce': 'p',
                               'SG': 'Pn-3'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'cubic',
                               'laue': 'm-3',
                               'class': 'm-3',
                               'ce': 'f',
                               'SG': 'Fm-3'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'cubic',
                               'laue': 'm-3',
                               'class': 'm-3',
                               'ce': 'f',
                               'SG': 'Fd-3'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'cubic',
                               'laue': 'm-3',
                               'class': 'm-3',
                               'ce': 'i',
                               'SG': 'Im-3'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'cubic',
                               'laue': 'm-3',
                               'class': 'm-3',
                               'ce': 'p',
                               'SG': 'Pa-3'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'cubic',
                               'laue': 'm-3m',
                               'class': '432',
                               'ce': 'i',
                               'SG': 'Ia-3'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'cubic',
                               'laue': 'm-3m',
                               'class': '432',
                               'ce': 'p',
                               'SG': 'P432'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'cubic',
                               'laue': 'm-3m',
                               'class': '432',
                               'ce': 'p',
                               'SG': 'P4232'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'cubic',
                               'laue': 'm-3m',
                               'class': '432',
                               'ce': 'f',
                               'SG': 'F432'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'cubic',
                               'laue': 'm-3m',
                               'class': '432',
                               'ce': 'f',
                               'SG': 'F4132'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'cubic',
                               'laue': 'm-3m',
                               'class': '432',
                               'ce': 'i',
                               'SG': 'I432'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'cubic',
                               'laue': 'm-3m',
                               'class': '432',
                               'ce': 'p',
                               'SG': 'P4332'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'cubic',
                               'laue': 'm-3m',
                               'class': '432',
                               'ce': 'p',
                               'SG': 'P4132'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'cubic',
                               'laue': 'm-3m',
                               'class': '432',
                               'ce': 'i',
                               'SG': 'I4132'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'cubic',
                               'laue': 'm-3m',
                               'class': '-43m',
                               'ce': 'p',
                               'SG': 'P-43m'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'cubic',
                               'laue': 'm-3m',
                               'class': '-43m',
                               'ce': 'f',
                               'SG': 'F-43m'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'cubic',
                               'laue': 'm-3m',
                               'class': '-43m',
                               'ce': 'i',
                               'SG': 'I-43m'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'cubic',
                               'laue': 'm-3m',
                               'class': '-43m',
                               'ce': 'p',
                               'SG': 'P-43n'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'cubic',
                               'laue': 'm-3m',
                               'class': '-43m',
                               'ce': 'f',
                               'SG': 'F-43c'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'cubic',
                               'laue': 'm-3m',
                               'class': '-43m',
                               'ce': 'i',
                               'SG': 'I-43d'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'cubic',
                               'laue': 'm-3m',
                               'class': 'm-3m',
                               'ce': 'p',
                               'SG': 'Pm-3m'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'cubic',
                               'laue': 'm-3m',
                               'class': 'm-3m',
                               'ce': 'p',
                               'SG': 'Pn-3n'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'cubic',
                               'laue': 'm-3m',
                               'class': 'm-3m',
                               'ce': 'p',
                               'SG': 'Pm-3n'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'cubic',
                               'laue': 'm-3m',
                               'class': 'm-3m',
                               'ce': 'p',
                               'SG': 'Pn-3m'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'cubic',
                               'laue': 'm-3m',
                               'class': 'm-3m',
                               'ce': 'f',
                               'SG': 'Fm-3m'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'cubic',
                               'laue': 'm-3m',
                               'class': 'm-3m',
                               'ce': 'f',
                               'SG': 'Fm-3c'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'cubic',
                               'laue': 'm-3m',
                               'class': 'm-3m',
                               'ce': 'f',
                               'SG': 'Fd-3m'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'cubic',
                               'laue': 'm-3m',
                               'class': 'm-3m',
                               'ce': 'f',
                               'SG': 'Fd-3c'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'cubic',
                               'laue': 'm-3m',
                               'class': 'm-3m',
                               'ce': 'i',
                               'SG': 'Im-3m'
                               })
    i += 1
    sgDict.loc[i] = pd.Series({'system': 'cubic',
                               'laue': 'm-3m',
                               'class': 'm-3m',
                               'ce': 'i',
                               'SG': 'Ia-3d'
                               })

    return sgDict


def dict2namespace(data_dict):
    """
    Recursively converts a dictionary and its internal dictionaries into an
    argparse.Namespace

    Parameters
    ----------
    data_dict : dict
        The input dictionary

    Return
    ------
    data_namespace : argparse.Namespace
        The output namespace
    """
    for k, v in data_dict.items():
        if isinstance(v, dict):
            data_dict[k] = dict2namespace(v)
        else:
            pass
    data_namespace = Namespace(**data_dict)

    return data_namespace


def load_yaml(path, append_config_dir=True):
    if append_config_dir:
        path = "configs/" + path
    yaml_path = Path(path)
    assert yaml_path.exists()
    assert yaml_path.suffix in {".yaml", ".yml"}
    with yaml_path.open("r") as f:
        target_dict = yaml.safe_load(f)

    return target_dict


def delete_from_dataframe(df, inds):
    df = df.drop(index=inds)
    if 'level_0' in df.columns:  # delete unwanted samples
        df = df.drop(columns='level_0')
    df = df.reset_index()

    return df


def compute_principal_axes_np(masses, coords):
    points = coords - coords.T.dot(masses)/np.sum(masses)
    x, y, z = points.T
    Ixx = np.sum(masses * (y ** 2 + z ** 2))
    Iyy = np.sum(masses * (x ** 2 + z ** 2))
    Izz = np.sum(masses * (x ** 2 + y ** 2))
    Ixy = -np.sum(masses * x * y)
    Iyz = -np.sum(masses * y * z)
    Ixz = -np.sum(masses * x * z)
    I = np.array([[Ixx, Ixy, Ixz], [Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]])  # inertial tensor
    Ipm, Ip = np.linalg.eig(I) # principal inertial tensor
    Ipm, Ip = np.real(Ipm), np.real(Ip)
    sort_inds = np.argsort(Ipm)
    Ipm = Ipm[sort_inds]
    Ip = Ip.T[sort_inds] # want eigenvectors to be sorted row-wise (rather than column-wise)

    # we want consistent directionality - set it against the CoG (Ipm always points towards CoG from CoM)
    direction = points.mean(axis=0) # CoM is 0 by construction, so we don't have to subtract it
    # if the CoG exactly == the CoM in any dimension, the molecule is symmetric on that axis, and we can arbitraily pick a side
    if any(direction) == 0:
        direction[direction==0] += 1 # arbitrarily set the positive side
    overlaps = np.sign(Ip.dot(direction)) # check if the principal components point towards or away from the CoG
    Ip = (Ip.T * overlaps).T # if the vectors have negative overlap, flip the direction

    return Ip, Ipm, I