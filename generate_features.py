# -*- coding: utf-8 -*-  
# Created by chenlu on May 2nd
import numpy as np
import pandas as pd
import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
import gc

def clicks_by_ip(X_total):
    start_time = time.time()
    # Count the number of clicks by ip
    ip_count = X_total.groupby(['ip']).size().rename('clicks_by_ip', inplace=True).reset_index()
    X_total = X_total.merge(ip_count, on='ip', how='left', sort=False)
    X_total['clicks_by_ip'] = X_total['clicks_by_ip'].astype('uint16')
    
    print(f'[{time.time() - start_time}] Finished to extract clicks_by_ip')
    return X_total

# Aggregation function
def rate_calculation(x):
    """Calculate the attributed rate. Scale by confidence"""
    rate = x.sum() / float(x.count())
    log_group = np.log(100000) # 1000 views -> 60% confidence, 100 views -> 40% confidence 
    conf = np.min([1, np.log(x.count()) / log_group])
    return rate * conf

def time_features(df):
    start_time = time.time()
    # Make some new features with click_time column
    df['day'] = df['click_time'].dt.day.astype('uint8')
    df['hour'] = df['click_time'].dt.hour.astype('uint8')
    df['minute'] = df['click_time'].dt.minute.astype('uint8')
    
    print(f'[{time.time() - start_time}] Finished to extract time_features')
    return df

def confidence_rate_feature(X_total, train):
    start_time = time.time()
    ATTRIBUTION_CATEGORIES = [        
        # V1 Features #
        ###############
        ['ip'], ['app'], ['device'], ['os'], ['channel'],

        # V2 Features #
        ###############
        ['app', 'channel'],
        ['app', 'os'],
        ['app', 'device'],

        # V3 Features #
        ###############
        ['channel', 'os'],
        ['channel', 'device'],
        ['os', 'device']
    ]

    # Find frequency of is_attributed for each unique value in column
    freqs = {}
    for cols in ATTRIBUTION_CATEGORIES:

        # New feature name
        new_feature = '_'.join(cols)+'_confRate'    

        # Perform the groupby
        group_object = train.groupby(cols)

        # Group sizes    
        group_sizes = group_object.size()
        log_group = np.log(100000) # 1000 views -> 60% confidence, 100 views -> 40% confidence 
        print(">> Calculating confidence-weighted rate for: {}.\n   Saving to: {}. Group Max /Mean / Median / Min: {} / {} / {} / {}"
              .format(
            cols, new_feature, 
            group_sizes.max(), 
            np.round(group_sizes.mean(), 2),
            np.round(group_sizes.median(), 2),
            group_sizes.min()
        ))

        # Perform the merge
        train = train.merge(
            group_object['is_attributed']. \
                apply(rate_calculation). \
                reset_index(). \
                rename( 
                    index=str,
                    columns={'is_attributed': new_feature}
                )[cols + [new_feature]],
            on=cols, how='left'
        )
    print(f'[{time.time() - start_time}] Finished to extract confidence_rate_feature')
    return train

def group_by_feature(train):
    start_time = time.time()
    # Define all the groupby transformations
    GROUPBY_AGGREGATIONS = [

        # V1 - GroupBy Features #
        #########################    
        # Variance in day, for ip-app-channel
        {'groupby': ['ip','app','channel'], 'select': 'day', 'agg': 'var'},
        # Variance in hour, for ip-app-os
        {'groupby': ['ip','app','os'], 'select': 'hour', 'agg': 'var'},
        # Variance in hour, for ip-day-channel
        {'groupby': ['ip','day','channel'], 'select': 'hour', 'agg': 'var'},
        # Count, for ip-day-hour
        {'groupby': ['ip','day','hour'], 'select': 'channel', 'agg': 'count'},
        # Count, for ip-app
        {'groupby': ['ip', 'app'], 'select': 'channel', 'agg': 'count'},        
        # Count, for ip-app-os
        {'groupby': ['ip', 'app', 'os'], 'select': 'channel', 'agg': 'count'},
        # Count, for ip-app-day-hour
        {'groupby': ['ip','app','day','hour'], 'select': 'channel', 'agg': 'count'},
        # Mean hour, for ip-app-channel
        {'groupby': ['ip','app','channel'], 'select': 'hour', 'agg': 'mean'}, 

        # V2 - GroupBy Features #
        #########################
        # Average clicks on app by distinct users; is it an app they return to?
        {'groupby': ['app'], 
         'select': 'ip', 
         'agg': lambda x: float(len(x)) / len(x.unique()), 
         'agg_name': 'AvgViewPerDistinct'
        },
        # How popular is the app or channel?
        {'groupby': ['app'], 'select': 'channel', 'agg': 'count'},
        {'groupby': ['channel'], 'select': 'app', 'agg': 'count'},

        # V3 - GroupBy Features                                              #
        # https://www.kaggle.com/bk0000/non-blending-lightgbm-model-lb-0-977 #
        ###################################################################### 
        {'groupby': ['ip'], 'select': 'channel', 'agg': 'nunique'}, 
        {'groupby': ['ip'], 'select': 'app', 'agg': 'nunique'}, 
        {'groupby': ['ip','day'], 'select': 'hour', 'agg': 'nunique'}, 
        {'groupby': ['ip','app'], 'select': 'os', 'agg': 'nunique'}, 
        {'groupby': ['ip'], 'select': 'device', 'agg': 'nunique'}, 
        {'groupby': ['app'], 'select': 'channel', 'agg': 'nunique'}, 
        {'groupby': ['ip', 'device', 'os'], 'select': 'app', 'agg': 'nunique'}, 
        {'groupby': ['ip','device','os'], 'select': 'app', 'agg': 'cumcount'}, 
        {'groupby': ['ip'], 'select': 'app', 'agg': 'cumcount'}, 
        {'groupby': ['ip'], 'select': 'os', 'agg': 'cumcount'}, 
        {'groupby': ['ip','day','channel'], 'select': 'hour', 'agg': 'var'}    
    ]

    # Apply all the groupby transformations
    for spec in GROUPBY_AGGREGATIONS:

        # Name of the aggregation we're applying
        agg_name = spec['agg_name'] if 'agg_name' in spec else spec['agg']

        # Name of new feature
        new_feature = '{}_{}_{}'.format('_'.join(spec['groupby']), agg_name, spec['select'])

        # Info
        print("Grouping by {}, and aggregating {} with {}".format(
            spec['groupby'], spec['select'], agg_name
        ))

        # Unique list of features to select
        all_features = list(set(spec['groupby'] + [spec['select']]))

        # Perform the groupby
        gp = train[all_features]. \
            groupby(spec['groupby'])[spec['select']]. \
            agg(spec['agg']). \
            reset_index(). \
            rename(index=str, columns={spec['select']: new_feature})

        # Merge back to X_total
        if 'cumcount' == spec['agg']:
            train[new_feature] = gp[0].values
        else:
            train = train.merge(gp, on=spec['groupby'], how='left')

        # Clear memory
        del gp
        
    print(f'[{time.time() - start_time}] Finished to extract group_by_feature')
    return train

def next_click_feature(train):
    start_time = time.time()
    GROUP_BY_NEXT_CLICKS = [

        # V1
        {'groupby': ['ip']},
        {'groupby': ['ip', 'app']},
        {'groupby': ['ip', 'channel']},
        {'groupby': ['ip', 'os']},

        # V3
        {'groupby': ['ip', 'app', 'device', 'os', 'channel']},
        {'groupby': ['ip', 'os', 'device']},
        {'groupby': ['ip', 'os', 'device', 'app']}
    ]

    # Calculate the time to next click for each group
    for spec in GROUP_BY_NEXT_CLICKS:

        # Name of new feature
        new_feature = '{}_nextClick'.format('_'.join(spec['groupby']))    

        # Unique list of features to select
        all_features = spec['groupby'] + ['click_time']

        # Run calculation
        print(f">> Grouping by {spec['groupby']}, and saving time to next click in: {new_feature}")
        train[new_feature] = train[all_features].groupby(spec['groupby']).click_time.transform(lambda x: x.diff().shift(-1)).dt.seconds
        
    print(f'[{time.time() - start_time}] Finished to extract next_click_feature')
    return train


def history_click_feature(train):
    start_time = time.time()
    HISTORY_CLICKS = {
        'identical_clicks': ['ip', 'app', 'device', 'os', 'channel'],
        'app_clicks': ['ip', 'app']
    }

    # Go through different group-by combinations
    for fname, fset in HISTORY_CLICKS.items():

        # Clicks in the past
        train['prev_'+fname] = train. \
            groupby(fset). \
            cumcount(). \
            rename('prev_'+fname)

        # Clicks in the future
        train['future_'+fname] = train.iloc[::-1]. \
            groupby(fset). \
            cumcount(). \
            rename('future_'+fname).iloc[::-1]

    # Count cumulative subsequent clicks
    print(f'[{time.time() - start_time}] Finished to extract history_click_feature')
    return train

def topic_feature(X_total):
    """
    categorical feature embedding by using LDA/NMF/LSA(PCA)
    """
    start_time = time.time()
    features = ['ip', 'app', 'device', 'os', 'channel']
    feat_len = len(features)
    n_components = 5
    
    for i in range(feat_len):
        for j in range(feat_len):
            if i == j:
                continue
            x = features[i]
            y = features[j]
            x_of_y = {}
            print(f">> generate topic embedding by {x}_of_{y}")
            for index, sample in X_total.iterrows():
                x_of_y.setdefault(sample[y], []).append(str(sample[x]))
            ys = list(x_of_y.keys())
            x_as_sentence = [' '.join(x_of_y[y]) for y in ys]
            x_as_matrix = CountVectorizer().fit_transform(x_as_sentence)
            print(f"[{time.time()-start_time}] generate x matrix done")
            # compute LDA topics of ys related to x
            lda = LatentDirichletAllocation(n_components=n_components,
                                            max_iter=3,
                                            learning_method='online',
                                            learning_offset=50.,
                                            random_state=0).fit_transform(x_as_matrix) # topics_of_ys
            lda_array = np.column_stack((np.array(ys), lda))
            print(f"[{time.time()-start_time}] LDA done")
            gc.collect()
            
            # computer NMF
            nmf = NMF(n_components=n_components,
                      random_state=1,
                      alpha=.1,
                      l1_ratio=.5).fit_transform(x_as_matrix)
            lda_nmf_array = np.column_stack((lda_array, nmf))
            print(f"[{time.time()-start_time}] NMF done")
            gc.collect()
            
            # compute LSA
            lsa = TruncatedSVD(n_components=n_components,
                               n_iter=3,
                               random_state=42).fit_transform(x_as_matrix)
            lda_nmf_lsa_array = np.column_stack((lda_nmf_array, lsa))
            print(f"[{time.time()-start_time}] LSA done")
            gc.collect()
            
            # add to X_total
            rename_dict = {0:y}
            t = ['LDA','NMF','LSA']
            for k in range(15):
                rename_dict[k+1] = f'{x}_of_{y}_{t[int(k/5)]}_{k%5+1}'
            temp = pd.DataFrame(data=lda_nmf_lsa_array).rename(index=str, columns=rename_dict)
            X_total = X_total.merge(temp, on=y, how='left', sort=False)
            print(f"[{time.time()-start_time}] merge done")
    X_total['ip'] = X_total['ip'].astype('uint32')
    X_total['app'] = X_total['app'].astype('uint16')
    X_total['device'] = X_total['device'].astype('uint16')
    X_total['os'] = X_total['os'].astype('uint16')
    X_total['channel'] = X_total['channel'].astype('uint16')
    
    print(f'[{time.time() - start_time}] Finished to extract topic_feature')
    return X_total