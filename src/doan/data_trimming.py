import pandas as pd

def trim_dataframe(df, max_samples, min_samples, column):
    df = df.copy()
    groups = df.groupby(column)
    trimmed_df = pd.DataFrame(columns=df.columns)
    for label in df[column].unique():
        group = groups.get_group(label)
        count = len(group)
        if count > max_samples:
            sampled_group = group.sample(n=max_samples, random_state=123, axis=0)
            trimmed_df = pd.concat([trimmed_df, sampled_group], axis=0)
        elif count >= min_samples:
            trimmed_df = pd.concat([trimmed_df, group], axis=0)
    print('after trimming, the maximum samples in any class is now ', max_samples, ' and the minimum samples in any class is ', min_samples)
    return trimmed_df