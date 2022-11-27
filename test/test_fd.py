from main.fd_violation import fd_violations

import pytest
import pandas as pd
from collections import Counter

def test_non_fd_1d():
    df = pd.DataFrame({'a':['A','B','C','D'],
                       'b':['AA','BB','CC','DD']})
    df = df.astype({'a': 'string', 'b': 'string'})
    groups = fd_violations(df, lhs='a', rhs='b')
    assert(groups == {})

def test_non_fd_1d_dup():
    df = pd.DataFrame({'a':['A','A','C','D'],
                       'b':['AA','AA','CC','DD']})
    df = df.astype({'a': 'string', 'b': 'string'})
    groups = fd_violations(df, lhs='a', rhs='b')
    assert(groups == {})

def test_fd_1d():
    df = pd.DataFrame({'a':['A','A','C','D'],
                       'b':['AA','BB','CC','DD']})
    df = df.astype({'a': 'string', 'b': 'string'})
    groups = fd_violations(df, lhs='a', rhs='b')
    # {'A': Counter({'AA': 1, 'BB': 1})}
    assert(len(groups.keys()) == 1)
    assert(list(groups.keys()) == list('A'))
    assert(list(groups.values())[0] == Counter(['AA', 'BB']))

def test_non_fd_2d():
    df = pd.DataFrame({'a':['A','B','C','D'],
                       'b':['AA','BB','CC','DD'],
                       'c':['AAA','BBB','CCC','DDD']})
    df = df.astype({'a': 'string', 'b': 'string',  'c': 'string'})
    groups = fd_violations(df, lhs=['a','b'], rhs='c')
    assert(groups == {})

def test_non_fd_2d_dup():
    df = pd.DataFrame({'a':['A','A','C','D'],
                       'b':['AA','AA','CC','DD'],
                       'c':['AAA','AAA','CCC','DDD']})
    df = df.astype({'a': 'string', 'b': 'string', 'c': 'string'})
    groups = fd_violations(df, lhs=['a','b'], rhs='c')
    assert(groups == {})

def test_fd_2d():
    df = pd.DataFrame({'a':['A','A','C','D'],
                       'b':['AA','AA','AA','AA'],
                       'c':['AAA','BBB','CCC','DDD']})
    df = df.astype({'a': 'string', 'b': 'string', 'c': 'string'})
    groups = fd_violations(df, lhs=['a','b'], rhs='c')
    assert(len(groups.keys()) == 1)
    assert(list(groups.keys()) == [tuple(['A','AA'])])
    assert(list(groups.values())[0] == Counter(['AAA', 'BBB']))


def test_invalid_df_error_1():
    df = pd.Series(['A','A','C','D'], dtype='string')
    with pytest.raises(ValueError) as e:
        fd_violations(df, lhs='a', rhs='b')
    assert str(e.value) == "df should be pd.DataFrame"

def test_invalid_df_error_2():
    df = ['A','A','C','D']
    with pytest.raises(ValueError) as e:
        fd_violations(df, lhs='a', rhs='b')
    assert str(e.value) == "df should be pd.DataFrame"

def test_invalid_dtype_rhs():
    df = pd.DataFrame({'a':['A','A','C','D'],
                       'b':[1,2,3,4]})
    df = df.astype({'a': 'string', 'b': 'float64'})
    with pytest.raises(ValueError) as e:
        fd_violations(df, lhs='a', rhs='b')
    assert str(e.value) == "rhs dtype should be string"

"""
def test_invalid_null_rhs():
    df = pd.DataFrame({'a':['A','A','C','D'],
                       'b':['AA','BB',None,'DD']})
    df = df.astype({'a': 'string', 'b': 'string'})
    with pytest.raises(ValueError) as e:
        fd_violations(df, lhs='a', rhs='b')
    assert str(e.value) == "rhs should not include null"
"""

def test_invalid_null_rhs():
    df = pd.DataFrame({'a':['A','A','C','D'],
                       'b':['AA',None,'CC','DD']})
    df = df.astype({'a': 'string', 'b': 'string'})
    groups = fd_violations(df, lhs='a', rhs='b')
    assert(len(groups.keys()) == 1)
    assert(list(groups.keys()) == list('A'))
    assert(list(groups.values())[0] == Counter(['AA', pd.NA]))

def test_invalid_dtype_lhs_1d():
    df = pd.DataFrame({'a':[1,2,3,4],
                       'b':['A','A','C','D']})
    df = df.astype({'a': 'float64', 'b': 'string'})
    with pytest.raises(ValueError) as e:
        fd_violations(df, lhs='a', rhs='b')
    assert str(e.value) == "lhs dtype should be string"

"""
def test_invalid_null_lhs_1d():
    df = pd.DataFrame({'a':['AA',None,'CC','DD'],
                       'b':['A','A','C','D']})
    df = df.astype({'a': 'string', 'b': 'string'})
    with pytest.raises(ValueError) as e:
        fd_violations(df, lhs='a', rhs='b')
    assert str(e.value) == "lhs should not include null"
"""

def test_invalid_null_lhs_1d():
    df = pd.DataFrame({'a':['AA',None,None,'DD'],
                       'b':['A','A','C','D']})
    df = df.astype({'a': 'string', 'b': 'string'})
    groups = fd_violations(df, lhs='a', rhs='b')
    assert(len(groups.keys()) == 1)
    assert(list(groups.keys()) == list(pd.Series([None], dtype='string')))
    assert(list(groups.values())[0] == Counter(['A', 'C']))


def test_invalid_dtype_lhs_2d_1():
    df = pd.DataFrame({'a':[1,2,3,4],
                       'b':['AA','AA','AA','AA'],
                       'c':['AAA','BBB','CCC','DDD']})
    df = df.astype({'a': 'float64', 'b': 'string', 'c': 'string'})
    with pytest.raises(ValueError) as e:
        fd_violations(df, lhs=['a','b'], rhs='c')
    assert str(e.value) == "lhs dtype should be string"

def test_invalid_dtype_lhs_2d_2():
    df = pd.DataFrame({'a':['A','B','C','D'],
                       'b':[1,2,3,4],
                       'c':['AAA','BBB','CCC','DDD']})
    df = df.astype({'a': 'string', 'b': 'float64', 'c': 'string'})
    with pytest.raises(ValueError) as e:
        fd_violations(df, lhs=['a','b'], rhs='c')
    assert str(e.value) == "lhs dtype should be string"

def test_invalid_dtype_lhs_2d_3():
    df = pd.DataFrame({'a':[1,2,3,4],
                       'b':[1,2,3,4],
                       'c':['AAA','BBB','CCC','DDD']})
    df = df.astype({'a': 'float64', 'b': 'float64', 'c': 'string'})
    with pytest.raises(ValueError) as e:
        fd_violations(df, lhs=['a','b'], rhs='c')
    assert str(e.value) == "lhs dtype should be string"

"""
def test_invalid_null_lhs_2d_1():
    df = pd.DataFrame({'a':['A','B','C','D'],
                       'b':['AA','BB','CC',None],
                       'c':['AAA','BBB','CCC','DDD']})
    df = df.astype({'a': 'string', 'b': 'string', 'c': 'string'})
    with pytest.raises(ValueError) as e:
        fd_violations(df, lhs=['a','b'], rhs='c')
    assert str(e.value) == "lhs should not include null"

def test_invalid_null_lhs_2d_2():
    df = pd.DataFrame({'a':['A','B','C',None],
                       'b':['AA','BB','CC','DD'],
                       'c':['AAA','BBB','CCC','DDD']})
    df = df.astype({'a': 'string', 'b': 'string', 'c': 'string'})
    with pytest.raises(ValueError) as e:
        fd_violations(df, lhs=['a','b'], rhs='c')
    assert str(e.value) == "lhs should not include null"
"""
def test_invalid_null_lhs_2d_1():
    df = pd.DataFrame({'a':['A','B','D','D'],
                       'b':['AA','BB',None,None],
                       'c':['AAA','BBB','CCC','DDD']})
    df = df.astype({'a': 'string', 'b': 'string', 'c': 'string'})
    groups = fd_violations(df, lhs=['a','b'], rhs='c')
    assert(len(groups.keys()) == 1)
    assert(list(groups.keys()) == [tuple(['D',pd.NA])])
    assert(list(groups.values())[0] == Counter(['CCC', 'DDD']))

def test_invalid_null_lhs_2d_2():
    df = pd.DataFrame({'a':['A','B',None,None],
                       'b':['AA','BB','DD','DD'],
                       'c':['AAA','BBB','CCC','DDD']})
    df = df.astype({'a': 'string', 'b': 'string', 'c': 'string'})
    groups = fd_violations(df, lhs=['a','b'], rhs='c')
    assert(len(groups.keys()) == 1)
    assert(list(groups.keys()) == [tuple([pd.NA,'DD'])])
    assert(list(groups.values())[0] == Counter(['CCC', 'DDD']))

def test_invalid_null_lhs_2d_3():
    df = pd.DataFrame({'a':['A','B',None,None],
                       'b':['AA','BB',None,None],
                       'c':['AAA','BBB','CCC','DDD']})
    df = df.astype({'a': 'string', 'b': 'string', 'c': 'string'})
    groups = fd_violations(df, lhs=['a','b'], rhs='c')
    assert(len(groups.keys()) == 1)
    assert(list(groups.keys()) == [tuple([pd.NA,pd.NA])])
    assert(list(groups.values())[0] == Counter(['CCC', 'DDD']))

def test_invalid_null_lhs_2d_4():
    df = pd.DataFrame({'a':['A','B','D','D'],
                       'b':['AA','BB','DD','DD'],
                       'c':['AAA','BBB','CCC',None]})
    df = df.astype({'a': 'string', 'b': 'string', 'c': 'string'})
    groups = fd_violations(df, lhs=['a','b'], rhs='c')
    assert(len(groups.keys()) == 1)
    assert(list(groups.keys()) == [tuple(['D','DD'])])
    assert(list(groups.values())[0] == Counter(['CCC', pd.NA]))

def test_invalid_rhs_instance_none():
    df = pd.DataFrame({'a':['AA','BB','CC','DD'],
                       'b':['A','A','C','D']})
    df = df.astype({'a': 'string', 'b': 'string'})
    with pytest.raises(ValueError) as e:
        fd_violations(df, lhs='a', rhs=None)
    assert str(e.value) == "rhs should be str"

def test_invalid_rhs_instance_list():
    df = pd.DataFrame({'a':['AA','BB','CC','DD'],
                       'b':['A','A','C','D']})
    df = df.astype({'a': 'string', 'b': 'string'})
    with pytest.raises(ValueError) as e:
        fd_violations(df, lhs='a', rhs=['a','b'])
    assert str(e.value) == "rhs should be str"

def test_invalid_lhs_instance_none():
    df = pd.DataFrame({'a':['AA','BB','CC','DD'],
                       'b':['A','A','C','D']})
    df = df.astype({'a': 'string', 'b': 'string'})
    with pytest.raises(ValueError) as e:
        fd_violations(df, lhs=None, rhs='a')
    assert str(e.value) == "lhs should be str or list"

def test_invalid_lhs_instance_dict():
    df = pd.DataFrame({'a':['AA','BB','CC','DD'],
                       'b':['A','A','C','D']})
    df = df.astype({'a': 'string', 'b': 'string'})
    with pytest.raises(ValueError) as e:
        fd_violations(df, lhs=dict({'a':1, 'b':2}), rhs='a')
    assert str(e.value) == "lhs should be str or list"

