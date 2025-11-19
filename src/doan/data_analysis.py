import numpy as np
import matplotlib.pyplot as plt

def analyze_dataset(train_df):
    classes = sorted(list(train_df['labels'].unique()))
    class_count = len(classes)
    groups = train_df.groupby('labels')
    print('{0:^30s} {1:^13s}'.format('CLASS', 'IMAGE COUNT'))
    countlist = []
    classlist = []
    for label in sorted(list(train_df['labels'].unique())):
        group = groups.get_group(label)
        countlist.append(len(group))
        classlist.append(label)
        print('{0:^30s} {1:^13s}'.format(label, str(len(group))))
    
    # Max/min classes
    max_value = np.max(countlist)
    max_index = countlist.index(max_value)
    max_class = classlist[max_index]
    min_value = np.min(countlist)
    min_index = countlist.index(min_value)
    min_class = classlist[min_index]
    print(max_class, ' has the most images= ', max_value, ' ', min_class, ' has the least images= ', min_value)
    
    # Avg height/width
    ht = 0
    wt = 0
    train_df_sample = train_df.sample(n=100, random_state=123, axis=0)
    for i in range(len(train_df_sample)):
        fpath = train_df_sample['filepaths'].iloc[i]
        img = plt.imread(fpath)
        shape = img.shape
        ht += shape[0]
        wt += shape[1]
    print('average height= ', ht//100, ' average width= ', wt//100, 'aspect ratio= ', ht/wt)
    
    return classes, class_count