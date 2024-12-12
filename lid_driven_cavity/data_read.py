import pandas as pd


def read_lid_driven_data(return_column_names=None):
    file_path = './lid_driven_cavity_data.txt'
    df = pd.read_csv(file_path, delim_whitespace=True, header=None, comment='%')

    column_names = [
        'x', 'y', 'spf.U @ Re=100', 'u @ Re=100', 'v @ Re=100', 'p @ Re=100',
        'spf.U @ Re=400', 'u @ Re=400', 'v @ Re=400', 'p @ Re=400',
        'spf.U @ Re=1000', 'u @ Re=1000', 'v @ Re=1000', 'p @ Re=1000',
        'spf.U @ Re=3200', 'u @ Re=3200', 'v @ Re=3200', 'p @ Re=3200',
        'spf.U @ Re=5000', 'u @ Re=5000', 'v @ Re=5000', 'p @ Re=5000',
        'spf.U @ Re=7500', 'u @ Re=7500', 'v @ Re=7500', 'p @ Re=7500',
        'spf.U @ Re=10000', 'u @ Re=10000', 'v @ Re=10000', 'p @ Re=10000'
    ]
    df.columns = column_names

    if return_column_names is not None:
        df = df[return_column_names]

    return df
