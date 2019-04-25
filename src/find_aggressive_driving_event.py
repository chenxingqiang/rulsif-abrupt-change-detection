# -*- coding: utf-8 -*-
# @Create Time    : 2019-03-27 12:33
# @Author  : Xingqiang Chen
# @Software: PyCharm

import os
import pandas as pd

from src.config import data_check_path, data_prod_path, \
    root_dir, event_length, n, data_feature_list


def find_file_dirs(file_dir):
    """
    :param file_dir:
    :return:
    """
    dirs_list = []
    paths = os.listdir(file_dir)

    for path in paths:
        dirs_list.append(os.path.join(file_dir, path))

    return dirs_list


def find_csv_file(csv_path, lenght_feature, dotname='csv'):
    """
    :param csv_path:
    :param dotname:
    :return:
    """
    L = []
    for root, dirs, files in os.walk(csv_path):
        for file in files:
            if file.split('.')[-1] == dotname:
                L.append(os.path.join(root, file))
    if len(L) < lenght_feature:
        print('FILE is NOT Enough!!' * 12)
        return False, sorted(L)
    elif len(L) >= lenght_feature:
        return True, sorted(L)


def init_path():
    """
    :return:
    """
    print(data_prod_path)
    print(data_check_path)

    name_app = os.path.join(root_dir, 'result_event_length_' + str(event_length))
    print(name_app)
    if not os.path.exists(name_app):
        os.makedirs(name_app)

    return name_app


def data_prepare(name_app):
    """
    :param name_app:
    :return:
    """

    length_feature = len(data_feature_list)
    alpha_name = ['alpha', '0.5']
    lambda_name = ['lambda', '1.5']

    dirs_list = find_file_dirs(data_prod_path)
    data_results = pd.DataFrame()

    for dir_name in dirs_list:
        state_flag, csv_file_path = find_csv_file(dir_name, length_feature)
        print(len(csv_file_path))

        if state_flag:
            csv_file_path = [x for x in csv_file_path if x.split('/')[-1].split('_')[-3] == alpha_name[-1]]
            print(csv_file_path)

            assert len(csv_file_path) >= len(data_feature_list)

            count_flag = 0
            data_app = pd.DataFrame()
            for data_path in csv_file_path:
                feature_name_app = data_path.split('.')[0].split('/')[-1].split('_')[1][0:3]
                Car_ID = data_path.split('.')[0].split('/')[-1].split('_')[0]
                data = pd.read_csv(data_path).iloc[:, 1:]
                data = data.replace('NOT_FULFILLMENT', 0.0)

                data['divergence_score'] = data['divergence_score'].astype(float)

                data.columns = ['Time', 'divergence_score_' + str(feature_name_app)]
                data['divergence_score_' + str(feature_name_app)] = data['divergence_score_' + str(
                    feature_name_app)].abs() * 1000

                if count_flag == 0:
                    data_app = data
                    count_flag = 1
                else:
                    data_app = pd.merge(data_app, data, on=['Time'], how='right')

            # print(data.head())

            data_orig = pd.read_csv(os.path.join(data_check_path, Car_ID, dir_name.split('/')[-1] + '.csv'))
            data_result = pd.merge(data_orig, data_app, on=['Time'], how='left')
            # standard of acd

            data_result['ds_total'] = data_result[['divergence_score_' + x[0:3] for x in data_feature_list]].sum(1)

            data_results = data_results.append(data_result)

            if len(data_results) >= event_length:
                top5_ds = find_top5(data_results)
                print('FINAL THRESHOLD IS:: ', top5_ds)

        else:
            continue

    data_results['is_acp'] = data_results['ds_total'].map(lambda x: 1 if x >= top5_ds else 0)

    data_results = data_results.reset_index().drop(['index'], axis=1)
    data_results.to_csv(os.path.join(name_app, 'drive_event_result.csv'))
    print('***** SAVING ORIGINAL EVENT DATA AS ' + os.path.join(name_app, 'drive_event_result.csv') + '  *****')
    save_event(data_results, name_app)


def find_top5(data_result):
    """
    :param data_result:
    :return:
    """
    assert len(data_result) >= event_length

    data_result = data_result.sort_values(by=['ds_total'], ascending=False)
    data_result = data_result.reset_index().drop(['index'], axis=1).fillna(0.0)

    top5 = int(0.05 * len(data_result))
    print('Event Length {0} Top 5 index is {1} ,and real data length of estimating is {2} '.format(event_length, top5,
                                                                                                   len(data_result)))
    top5_ds = data_result.ds_total.values[top5]
    print('top5 divergence_score ', top5_ds)

    return top5_ds


def save_event(data, name_app):
    """
    :param data:
    :param name_app:
    :return:
    """
    data_re = data[data.is_acp == 1]
    events_index = [[int(x) - (n - 1), int(x) + 1] for x in list(data_re.index)]
    event_df = pd.DataFrame(events_index)
    event_df.columns = ['start_index', 'end_index']

    event_df.to_csv(os.path.join(name_app, 'drive_event_index.csv'))
    print('*****  SAVING DRIVE EVENT INDEX AS ' + os.path.join(name_app, 'drive_event_index.csv') + ' *****')


def find_event():
    """
    :return:
    """
    name_app = init_path()
    data_prepare(name_app)


if __name__ == '__main__':
    find_event()
