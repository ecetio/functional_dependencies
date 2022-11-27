from __future__ import annotations
from collections import defaultdict, Counter
from typing import Union, List

import pandas as pd



def fd_violations(
    df: pd.DataFrame, lhs: Union[str, List[str]], rhs: str
) -> dict:
    """ Checks for violations of a functional dependency in the given data frame."""
    def duplicates(meta):
        return len(meta) > 1

    # DataFrameの型を確認
    if not isinstance(df, pd.DataFrame):
        raise ValueError('df should be pd.DataFrame')

    if not isinstance(rhs, str):
        raise ValueError('rhs should be str')

    if (not isinstance(lhs, str)) and (not isinstance(lhs, list)):
        raise ValueError('lhs should be str or list')


    # 列の型はstringのみ
    if not df[rhs].dtypes.name in ['string']:
        raise ValueError('rhs dtype should be string')
    #if df[rhs].isnull().any():
    #    raise ValueError('rhs should not include null')

    if isinstance(lhs, str):
        if not df[lhs].dtypes.name in ['string']:
            raise ValueError('lhs dtype should be string')
        #if df[lhs].isnull().any():
        #    raise ValueError('lhs should not include null')
    else:
        for col in lhs:
            if not df[col].dtypes.name in ['string']:
                raise ValueError('lhs dtype should be string')
            #if df[col].isnull().any():
            #    raise ValueError('lhs should not include null')


    df_determinant = df[lhs]
    df_dependent = df[rhs]

    determinant = df_determinant.to_numpy().tolist()
    dependent = df_dependent.to_numpy().tolist()
    #print(determinant)
    #print(dependent)

    groups = dict()
    meta = dict()
    for index, values in enumerate(zip(determinant, dependent)):
        value = values[0]  # determinant value: keys
        if isinstance(value, list):
            value = tuple(value)
        if value not in groups:
            groups[value] = list()
            meta[value] = Counter()
        groups[value].append(index)

        meta_value = values[1]  # dependent value: meta

        counter = Counter([meta_value])
        meta[value] += counter

        
    grouping = dict()
    for key, rows in groups.items():
        if duplicates(meta[key]):
            #if key in grouping:
            #    raise ValueError('duplicate key {}'.format(key))
            grouping[key] = meta[key]

    return grouping


