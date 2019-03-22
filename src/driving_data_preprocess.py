import pandas as pd
import os
from src.config import data_path,data_check_path,data_name


def Less_Than(seq):
    """
    :param seq:
    :return:
    """
    assert len(seq) >= 2
    for index, value in enumerate(seq[:-1]):
        if value != seq[index + 1] - 100:
            return False
    return True


def detectSeqChange(seq):
    """
    :param seq:
    :return:
    """
    scp = []
    for index, value in enumerate(seq[:-1]):
        if value != seq[index + 1] - 100:
            scp.append(int(len(seq)) - int(index))
    return [0] + sorted(scp) + [len(seq)]


def preprocess(data_name):
    """
    :param data_name:
    :return:
    """
    for name in data_name:
        data = pd.read_csv(os.path.join(data_path,name), index_col=None, low_memory=False)
        print("Driving Car ID Set:", set(data.ID))
        data = data.reset_index().drop(['index'], axis=1)
        data.columns = ['Car_ID', 'Time', 'Car_Orientation', 'Pitch_Rate', 'Roll_Rate', 'Acceleration', 'Velocity',
                        'Steering_Wheel_Angle', 'Yaw_Rate']
        print(len(data[data.Time == 0]))

        sep_list = list(data[data.Time == 0].index)

        if len(sep_list) == 0:
            sep_list = [0, len(data)]

        elif len(sep_list) > 0 and sep_list[0] > 0:
            if sep_list[-1] == len(data) - 1:
                sep_list = [0] + sep_list + [len(data) + 1]
            else:
                sep_list = [0] + sep_list + [len(data)]

        # print(sep_list)
        # print(data.info())

        print(' ####' * 12 + ' CHECKING LIST !!! ' + 12 * '#### ')
        file_name = name.strip('.csv')
        save_path = os.path.join(data_check_path, file_name)
        print(save_path)

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        for i in range(1, len(sep_list)):

            seq = list(data.Time[sep_list[i - 1]:sep_list[i]].astype(int))
            if len(seq) > 151:

                flag = Less_Than(seq)
                if flag:
                    data.iloc[sep_list[i - 1]:sep_list[i], :].to_csv(os.path.join(
                        save_path,'OrderRight_' + name.strip('.csv') + '_' + str(i).zfill(4) + '.csv'))
                    print('save ok for {} '.format(i))

                if not flag:
                    scp = detectSeqChange(seq)
                    print('SCP:', scp)
                    for i in range(1, len(scp)):
                        print(scp[i - 1], scp[i])
                        if scp[i] > 151:
                            print('SCP WORKING', scp)
                            data.iloc[sep_list[i - 1] + scp[i - 1]:sep_list[i - 1] + scp[i], :].to_csv(os.path.join(
                                save_path , 'OrderRight_' + name.strip('.csv') + '_' + str(i).zfill(
                                    4) + '_scpindex_' + str(sep_list[i - 1] + scp[i]) + '.csv'))

                        else:
                            print('seq is less length at starting')
                            data.iloc[sep_list[i - 1] + scp[i - 1]:sep_list[i - 1] + scp[i], :].to_csv(os.path.join(
                                save_path, 'LessLength_' + name.strip('.csv') + '_' + str(i).zfill(
                                    4) + '_scpindex_' + str(sep_list[i - 1] + scp[i]) + '.csv'))

            else:
                if Less_Than(list(seq)):
                    data.iloc[sep_list[i - 1]:sep_list[i], :].to_csv(os.path.join(
                        save_path , 'LessLength' + name.strip('.csv') + '_' + str(i).zfill(4) + '.csv'))
                    print('LessLength!! {} '.format(i))
                else:
                    data.iloc[sep_list[i - 1]:sep_list[i], :].to_csv(os.path.join(
                        save_path , 'LessLength_' + 'OrderError_' + name.strip('.csv') + '_' + str(i).zfill(4) + '.csv'))
                    print('OrderError!! LessLength!! for {} '.format(i) * 8)


def apply_preprocess():
    """
    :return:
    """
    # data load
    preprocess(data_name)


if __name__ == '__main__':
    apply_preprocess()
