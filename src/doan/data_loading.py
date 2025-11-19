import pandas as pd
import os

def create_dataframe(sdir):
    classlist = sorted(os.listdir(sdir))
    filepaths = []
    labels = []
    for klass in classlist:
        classpath = os.path.join(sdir, klass)
        flist = sorted(os.listdir(classpath))
        for f in flist:
            fpath = os.path.join(classpath, f)
            filepaths.append(fpath)
            labels.append(klass)
    Fseries = pd.Series(filepaths, name='filepaths')
    Lseries = pd.Series(labels, name='labels')
    df = pd.concat([Fseries, Lseries], axis=1)
    return df