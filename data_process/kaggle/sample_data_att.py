import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
np.set_printoptions(threshold=np.inf)


if __name__ == '__main__':
    # for kaggle
    data_directory = 'data'
    df1 = pd.read_csv(os.path.join(data_directory, '2019-Oct.csv'), header=None, skiprows=1)
    df2 = pd.read_csv(os.path.join(data_directory, '2019-Nov.csv'), header=None, skiprows=1)
    df3 = pd.read_csv(os.path.join(data_directory, '2019-Dec.csv'), header=None, skiprows=1)
    df4 = pd.read_csv(os.path.join(data_directory, '2020-Jan.csv'), header=None, skiprows=1)
    df5 = pd.read_csv(os.path.join(data_directory, '2020-Feb.csv'), header=None, skiprows=1)
    df = pd.concat([df1, df2, df3, df4, df5], ignore_index=True)
    
    df.columns = ['event_time', 'event_type', 'product_id', 'category_id', 'category_code', 'brand', 'price', 'user_id', 'user_session']
    # where event_type concludes 'view', 'cart', 'remove_from_cart', 'purchase'.
    
    df['timestamp'] = pd.to_datetime(df['event_time'])
    df['timestamp'] = df['timestamp'].astype(int) // 10**9
    
    print("#users:{},#items:{},#category:{},#price:{},#brand:{},#view:{},#remove_from_cart:{},#cart:{},#purchase:{}".format(
        len(df['user_id'].unique()),
        len(df['product_id'].unique()),
        len(df['category_id'].unique()),
        len(df['price'].unique()),
        len(df['brand'].unique()),
        len(df[df['event_type'] == 'view']),
        len(df[df['event_type'] == 'remove_from_cart']),
        len(df[df['event_type'] == 'cart']),
        len(df[df['event_type'] == 'purchase']),
    ))
    
    df['action_type'] = ''
    df['action_type'].loc[df['event_type'] == 'purchase'] = 0   # purchase
    df['action_type'].loc[df['event_type'] == 'remove_from_cart'] = 1   # remove from cart
    df['action_type'].loc[df['event_type'] == 'cart'] = 2   # cart
    df['action_type'].loc[df['event_type'] == 'view'] = 3   # view
    print("#users:{},#items:{},#category:{},#price:{},#brand:{},#view:{},#remove_from_cart:{},#cart:{},#purchase:{}".format(
        len(df['user_id'].unique()),
        len(df['product_id'].unique()),
        len(df['category_id'].unique()),
        len(df['price'].unique()),
        len(df['brand'].unique()),
        len(df[df['action_type'] == 3]),
        len(df[df['action_type'] == 1]),
        len(df[df['action_type'] == 2]),
        len(df[df['action_type'] == 0]),
    ))
    
    # without price
    df = df.drop(columns=['event_time', 'event_type', 'category_code', 'user_session', 'price'])
    
    df.rename(columns={'product_id': 'item_id', 'category_id': 'cat_id', 'brand': 'brand_id'}, inplace=True)
    # swap columns
    df = df[['user_id', 'item_id', 'cat_id', 'brand_id', 'timestamp', 'action_type']]
    
    df.sort_values(['user_id', 'timestamp'], inplace=True)
    print("#users:{},#items:{},#category:{},#brand:{},#view:{},#remove_from_cart:{},#cart:{},#purchase:{}".format(
        len(df['user_id'].unique()),
        len(df['item_id'].unique()),
        len(df['cat_id'].unique()),
        len(df['brand_id'].unique()),
        len(df[df['action_type'] == 3]),
        len(df[df['action_type'] == 1]),
        len(df[df['action_type'] == 2]),
        len(df[df['action_type'] == 0]),
    ))
    
    df = df.drop(df[df['action_type'].isin([1])].index)
    df.drop_duplicates(subset=['user_id', 'item_id', 'action_type'], keep='first', inplace=True)

    print("#users:{},#items:{},#category:{},#brand:{},#view:{},#remove_from_cart:{},#cart:{},#purchase:{}".format(
        len(df['user_id'].unique()),
        len(df['item_id'].unique()),
        len(df['cat_id'].unique()),
        len(df['brand_id'].unique()),
        len(df[df['action_type'] == 3]),
        len(df[df['action_type'] == 1]),
        len(df[df['action_type'] == 2]),
        len(df[df['action_type'] == 0]),
    ))

    df = df.fillna('-1')
    print(df.isnull().sum())
    print("#users:{},#items:{},#category:{},#brand:{},#view:{},#remove_from_cart:{},#cart:{},#purchase:{}".format(
        len(df['user_id'].unique()),
        len(df['item_id'].unique()),
        len(df['cat_id'].unique()),
        len(df['brand_id'].unique()),
        len(df[df['action_type'] == 3]),
        len(df[df['action_type'] == 1]),
        len(df[df['action_type'] == 2]),
        len(df[df['action_type'] == 0]),
    ))
    
    # delete buy less than 20
    df['is_buy'] = df['action_type'].map(lambda x: 1 if x == 0 else 0)
    df['valid_item'] = df.item_id.map(df.groupby('item_id')['is_buy'].sum() >= 20)
    df = df.loc[df.valid_item].drop('valid_item', axis=1)
    print("#users:{},#items:{},#category:{},#brand:{},#view:{},#remove_from_cart:{},#cart:{},#purchase:{}".format(
        len(df['user_id'].unique()),
        len(df['item_id'].unique()),
        len(df['cat_id'].unique()),
        len(df['brand_id'].unique()),
        len(df[df['action_type'] == 3]),
        len(df[df['action_type'] == 1]),
        len(df[df['action_type'] == 2]),
        len(df[df['action_type'] == 0]),
    ))
    # delete users without purchase
    df = df[df.user_id.isin(df[df['action_type'] == 0].user_id.unique())]
    print("#users:{},#items:{},#category:{},#brand:{},#view:{},#remove_from_cart:{},#cart:{},#purchase:{}".format(
        len(df['user_id'].unique()),
        len(df['item_id'].unique()),
        len(df['cat_id'].unique()),
        len(df['brand_id'].unique()),
        len(df[df['action_type'] == 3]),
        len(df[df['action_type'] == 1]),
        len(df[df['action_type'] == 2]),
        len(df[df['action_type'] == 0]),
    ))
    # delete cold-start users with less than 10 purchases
    df['valid_session'] = df.user_id.map(df[df['action_type'] == 0].groupby('user_id')['item_id'].size() >= 10)
    df = df.loc[df.valid_session].drop('valid_session', axis=1)
    print("#users:{},#items:{},#category:{},#brand:{},#view:{},#remove_from_cart:{},#cart:{},#purchase:{}".format(
        len(df['user_id'].unique()),
        len(df['item_id'].unique()),
        len(df['cat_id'].unique()),
        len(df['brand_id'].unique()),
        len(df[df['action_type'] == 3]),
        len(df[df['action_type'] == 1]),
        len(df[df['action_type'] == 2]),
        len(df[df['action_type'] == 0]),
    ))

    df = df.reset_index(drop=True)
    df['idx'] = df.index
    # last_two_buy is the last second buy in df form
    last_two_buy = df[df['action_type'] == 0].groupby(['user_id']).tail(2).groupby(['user_id']).head(1)
    last_one_buy = df[df['action_type'] == 0].groupby(['user_id']).tail(2).groupby(['user_id']).tail(1)
    last_two_buy_map = dict(zip(last_two_buy['user_id'].values.tolist(), last_two_buy['idx'].values.tolist()))
    last_one_buy_map = dict(zip(last_one_buy['user_id'].values.tolist(), last_one_buy['idx'].values.tolist()))

    df['last_two_buy_idx'] = df['user_id'].map(last_two_buy_map)
    df['last_one_buy_idx'] = df['user_id'].map(last_one_buy_map)

    # delete the records of valid and test items
    last_two_buy_item_map = dict(
        zip(last_two_buy['user_id'].values.tolist(), last_two_buy['item_id'].values.tolist()))
    last_one_buy_item_map = dict(
        zip(last_one_buy['user_id'].values.tolist(), last_one_buy['item_id'].values.tolist()))
    df['last_two_buy_item'] = df['user_id'].map(last_two_buy_item_map)
    df['last_one_buy_item'] = df['user_id'].map(last_one_buy_item_map)
    df = df.groupby(['user_id']).apply(
        lambda x: x[~((x['action_type'] != 0) & (x['item_id'] == x['last_two_buy_item']))]).reset_index(drop=True)
    print(df)
    df = df.groupby(['user_id']).apply(
        lambda x: x[~((x['action_type'] != 0) & (x['item_id'] == x['last_one_buy_item']))]).reset_index(drop=True)
    print(df)
    ###################################################
    df_between_val_test = df.groupby(['user_id']).apply(
        lambda x: x[((x['action_type'] != 0) & (x['idx'] > x['last_two_buy_idx']) & (x['idx'] < x['last_one_buy_idx']))]).reset_index(drop=True)
    print(df_between_val_test)
    df = df.groupby(['user_id']).apply(lambda x: x[~((x['action_type'] != 0) & (x['idx'] > x['last_two_buy_idx']))]).reset_index(drop=True)
    print(df)
    print("#users:{},#items:{},#category:{},#brand:{},#view:{},#remove_from_cart:{},#cart:{},#purchase:{}".format(
        len(df['user_id'].unique()),
        len(df['item_id'].unique()),
        len(df['cat_id'].unique()),
        len(df['brand_id'].unique()),
        len(df[df['action_type'] == 3]),
        len(df[df['action_type'] == 1]),
        len(df[df['action_type'] == 2]),
        len(df[df['action_type'] == 0]),
    ))
    df = df.groupby(['user_id']).tail(53)
    print("#users:{},#items:{},#category:{},#brand:{},#view:{},#remove_from_cart:{},#cart:{},#purchase:{}".format(
        len(df['user_id'].unique()),
        len(df['item_id'].unique()),
        len(df['cat_id'].unique()),
        len(df['brand_id'].unique()),
        len(df[df['action_type'] == 3]),
        len(df[df['action_type'] == 1]),
        len(df[df['action_type'] == 2]),
        len(df[df['action_type'] == 0]),
    ))
    # original data: 0 is click, 1 is cart, 2 is purchase and 3 is favourite.
    # we reindex into: 0 is purchase, 1 is cart, 2 is favorite, 3 is view
    df_len = df.shape[0]
    df = pd.concat([df, df_between_val_test], axis=0)

    # purchase:0; cart:1; favorite:2; click:3; 

    print(df.groupby(['user_id'])['item_id'].size().mean())
    print(df.columns)
    df = df.drop(columns=['idx', 'last_two_buy_idx', 'last_one_buy_idx', 'is_buy', 'last_two_buy_item', 'last_one_buy_item'])
    print(df.columns)

    item_encoder = LabelEncoder()
    user_encoder = LabelEncoder()
    
    # for attributes
    cat_encoder = LabelEncoder()
    # price_encoder = LabelEncoder()
    brand_encoder = LabelEncoder()
    
    df['user_id'] = user_encoder.fit_transform(df.user_id) + 1
    df['item_id'] = item_encoder.fit_transform(df.item_id) + 1
    
    # for attributes
    df['cat_id'] = cat_encoder.fit_transform(df.cat_id) + 1
    # df['price_id'] = price_encoder.fit_transform(df.price_id) + 1
    df['brand_id'] = brand_encoder.fit_transform(df.brand_id) + 1
    
    
    df_between_val_test = df.iloc[df_len:]
    df = df.iloc[:df_len]
    print(df)
    # ========================================================
    # split data into test set, valid set and train set,
    # adopting the leave-one-out evaluation for next-item recommendation task
    # ========================================
    # obtain possible records in test set
    df_test = df.groupby(['user_id']).tail(1)
    df.drop(df_test.index, axis='index', inplace=True)

    # ========================================
    # obtain possible records in valid set
    df_valid = df.groupby(['user_id']).tail(1)
    df.drop(df_valid.index, axis='index', inplace=True)

    # ========================================
    # drop cold-start items in valid set and test set
    df_valid = df_valid[df_valid.item_id.isin(df.item_id)]
    df_test = df_test[df_test.user_id.isin(df_valid.user_id) & (
        df_test.item_id.isin(df.item_id) | df_test.item_id.isin(df_valid.item_id))]

    processed_file_prefix = "processed_data_att/kaggle_"
    # output data file
    df_valid.to_csv(processed_file_prefix + "valid.csv", header=False, index=False)
    df_test.to_csv(processed_file_prefix + "test.csv", header=False, index=False)
    df.to_csv(processed_file_prefix + "train.csv", header=False, index=False)
    df_between_val_test.to_csv(processed_file_prefix + "between_val_test.csv", header=False, index=False)

    # ========================================================
    # For each user, randomly sample some negative items,
    # and rank these items with the ground-truth item when testing or validation
    df_concat = pd.concat([df, df_valid, df_test, df_between_val_test], axis='index')
    print(df_concat)
    sr_user2items = df_concat.groupby(['user_id']).item_id.unique()
    print(sr_user2items)
    df_negative = pd.DataFrame({'user_id': df_concat.user_id.unique()})
    print(df_negative)
    # ========================================
    # sample according to popularity

    sr_item2pop = df_concat.item_id.value_counts(sort=True, ascending=False)
    arr_item = sr_item2pop.index.values
    arr_pop = sr_item2pop.values


    def get_negative_sample(pos):
        neg_idx = ~np.in1d(arr_item, pos)
        neg_item = arr_item[neg_idx]
        neg_pop = arr_pop[neg_idx]
        neg_pop = neg_pop / neg_pop.sum()

        return np.random.choice(neg_item, size=100, replace=False, p=neg_pop)
        # return np.random.choice(neg_item, size=100, replace=False)


    arr_sample = df_negative.user_id.apply(
        lambda x: get_negative_sample(sr_user2items[x])).values

    # output negative data
    df_negative = pd.concat([df_negative, pd.DataFrame(list(arr_sample))], axis='columns')
    df_negative.to_csv(processed_file_prefix + "negative.csv", header=False, index=False)
    # df_negative.to_csv(processed_file_prefix + "negative_temp.csv", header=False, index=False)
