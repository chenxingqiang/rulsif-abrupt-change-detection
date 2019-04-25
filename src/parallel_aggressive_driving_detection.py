# -*- coding: utf-8 -*-
# @Create Time    : 2019-03-27 12:33
# @Author  : Xingqiang Chen
# @Software: PyCharm

import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait

import numpy as np
import pandas as pd

from src.config import data_check_path, data_prod_path, \
    settings, before_Times, n, MPI, restart, start, end, data_feature_list
from src.rulsif import RULSIF

_executor_pool = ThreadPoolExecutor(max_workers=32)


def calculate_divergence_score(X_reference, X_test, settings):
    """
    :param X_reference:
    :param X_test:
    :param settings:
    :return:
    """
    estimator = RULSIF(settings=settings)

    # Train the model
    estimator.train(X_reference, X_test)

    divergence_score = estimator.apply(X_reference, X_test)
    options = {'--debug': 1}
    # estimator.show(displayName='try',options=options)

    return divergence_score


def condition_time_series(condition, before_Times, feature_name):
    """
    :param condition:
    :param before_Times:
    :param feature_name:
    :return:
    """

    Times = sorted(list(set(condition.Time)), reverse=False)
    con_all_day = condition[['Car_ID', 'Time', feature_name]].drop_duplicates()
    df_test = con_all_day.pivot_table(index=['Car_ID'], columns=['Time'], values=[feature_name])
    con_day_n = df_test[feature_name].sort_index(axis=1, ascending=True).reset_index()
    Before_con_cols = ['before_' + feature_name + '_' + str(i).zfill(2) for i in range(before_Times)]

    before_data = pd.DataFrame()

    for i in range(1, len(Times[0:-before_Times]) + 1):
        before = con_day_n[con_day_n.columns[-before_Times - i: - i]]
        before.columns = Before_con_cols[0:before_Times]
        before_title = con_day_n['Car_ID']
        before_end = pd.concat([before_title, before], axis=1)
        before_end.insert(1, 'Time', Times[-before_Times - i])

        # drop duplicates
        before_end = before_end.drop_duplicates()
        before_data = pd.concat([before_data, before_end], axis=0)

    before_last = con_day_n[con_day_n.columns[-before_Times:]]
    before_last.columns = Before_con_cols[0:before_Times]

    before_last.insert(0, 'Car_ID', before_data.iloc[1, 0])
    before_last.insert(1, 'Time', Times[-before_Times])

    before_data = before_data.append(before_last)

    before_data = before_data.sort_values(by=['Time'])
    before_data = before_data.reset_index().drop('index', axis=1)

    return before_data


class task(object):

    def __init__(self, data_name_list):
        """
        :param data_name_list:
        """
        self.data_name_list = data_name_list
        self.data_prod_path = data_prod_path

    def calculation_all(self, data, save_path):
        """
        :param data:
        :param save_path:
        :return:
        """

        for feature_name in data_feature_list:
            condition = data[['Car_ID', 'Time', feature_name]]
            Hankel_Seq = condition_time_series(condition, before_Times, feature_name)
            car_ID = str(int(Hankel_Seq.Car_ID[0]))

            result = self.re(Hankel_Seq, feature_name, car_ID)
            print('Car ID: ', car_ID, '  ', 'Feature Name: ', feature_name, 'finished')

            alpha = str(settings['--alpha'])
            lam = str(settings['--lambda'])
            save_data = pd.DataFrame(result)

            save_data.columns = ['Time', 'divergence_score']
            save_data.to_csv(os.path.join(save_path, car_ID + '_' + feature_name + '_alpha_' + alpha + '_lambda_' + lam + '.csv'))

        # del Hankel_Seq,condition,save_data

    @staticmethod
    def re(Hankel_Seq, feature_name, car_ID):
        """
        :param Hankel_Seq:
        :param feature_name:
        :param car_ID:
        :return:
        """
        result = []
        Hankel_Seq = Hankel_Seq.replace(0, -1)

        for i in range(0, len(Hankel_Seq.Time) - n):
            Y_ref = Hankel_Seq.iloc[i:i + n, 2:].values
            Y_tes = Hankel_Seq.iloc[i + 1:i + n + 1, 2:].values

            a = len(set(np.concatenate(Y_ref.tolist(), axis=0)))
            b = len(set(np.concatenate(Y_tes.tolist(), axis=0)))

            if a > 1 or b > 1:
                divergence_score = calculate_divergence_score(Y_ref, Y_tes, settings)
                result.append([Hankel_Seq.iloc[i + n, 1], divergence_score])
            else:
                print('TEST NOT OK: ', 'Car ID: ', car_ID, '  ', 'Feature Name: ', feature_name, '::',
                      'Test Time Interval:', Hankel_Seq.iloc[i + 1, 1], Hankel_Seq.iloc[i + n, 1])
                result.append([Hankel_Seq.iloc[i + n, 1], 'NOT_FULFILLMENT'])

        return result

    def apply(self):
        """
        :return:
        """

        for name in self.data_name_list:
            print(name)
            file_name = name.split('/')[-1].strip('.csv')
            save_path = os.path.join(self.data_prod_path,file_name)
            print(save_path)
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            data = pd.read_csv(name, low_memory=False).iloc[:, 1:]
            data.columns = ['Car_ID', 'Time', 'Pitch_Rate', 'Roll_Rate', 'Acceleration', 'Car_Orientation',
                            'Velocity', 'Steering_Wheel_Angle', 'Yaw_Rate']

            print("Driving Car ID Set:", set(data.Car_ID))
            data = data.reset_index().drop(['index'], axis=1)

            self.calculation_all(data, save_path)


def find_name(file_dir, dotname='csv', fileType='OrderRight'):
    """
    :param file_dir:
    :param dotname:
    :param fileType:
    :return:
    """
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if file.split('.')[-1] == dotname and file.split('_')[0] == fileType:
                L.append(os.path.join(root, file))
    return sorted(L)


def MPI_task(tasker):
    tasker.apply()


def check_name(file_dir, dotname='csv', fileType='OrderRight'):
    """
    :param file_dir:
    :param dotname:
    :param fileType:
    :return:
    """

    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:

            if file.split('.')[-1] == dotname and file[-24:-4] == fileType:

                L.append(os.path.join(root, file))
    return L


def apply_detection():
    global data_name, data_name_update

    data_name = find_name(data_check_path, dotname='csv', fileType='OrderRight')
    print('The Total Number of Files: ', len(data_name))

    if restart:
        data_prod = check_name(data_prod_path, dotname='csv', fileType='alpha_0.5_lambda_1.5')
        data_name_set = set( data_name )
        data_prod_list = [x.split('/')[-2] for x in data_prod]
        data_prod_counter = Counter(data_prod_list)

        exist_name = []
        for item in data_prod_counter.keys():
            if data_prod_counter[item]>=4:
                exist_name.append(os.path.join(data_prod_path,item.split('_')[1],item+'.csv'))

        exist_name_set = set(exist_name)

        print(len(exist_name_set))
        print(len(data_name_set))
        data_name_update = list(data_name_set.difference(exist_name_set))
        print(len(data_name_update))

        if MPI:
            data_name_run = data_name_update[start:end]
            future_objs = []
            num = len(data_name_run)
            for i in range(num):
                future_objs.append(task([data_name[i]]))

            future_object = []
            for i in range(num):
                obj = _executor_pool.submit(MPI_task, future_objs[i])
                future_object.append(obj)
            # wait for all job finished!
            wait(future_object)

        else:
            data_name_list = data_name_update[start:end]
            task(data_name_list)
    else:
        if MPI:
            data_name_run = data_name[start:end]
            future_objs = []
            num = len(data_name_run)
            for i in range(num):
                future_objs.append(task([data_name[i]]))

            future_object = []
            for i in range(num):
                obj = _executor_pool.submit(MPI_task, future_objs[i])
                future_object.append(obj)
            # wait for all job finished!
            wait(future_object)

        else:
            data_name_list = data_name_update[start:end]
            task(data_name_list)


if __name__ == '__main__':
    apply_detection()
