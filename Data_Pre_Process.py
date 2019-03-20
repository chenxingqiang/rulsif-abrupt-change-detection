import pandas as pd
import os


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


def pre_process(data_name, data_path, data_save_path):
    """
    :param data_name:
    :param data_path:
    :param data_save_path:
    :return:
    """
    for name in data_name:
        data = pd.read_csv(data_path + name, index_col=None, low_memory=False)
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
        save_path = data_save_path + file_name + '/'
        print(save_path)

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        for i in range(1, len(sep_list)):

            seq = list(data.Time[sep_list[i - 1]:sep_list[i]].astype(int))
            if len(seq) > 151:

                flag = Less_Than(seq)
                if flag:
                    data.iloc[sep_list[i - 1]:sep_list[i], :].to_csv(
                        save_path + 'OrderRight_' + name.strip('.csv') + '_' + str(i).zfill(4) + '.csv')
                    print('save ok for {} '.format(i))

                if not flag:
                    scp = detectSeqChange(seq)
                    print('SCP:', scp)
                    for i in range(1, len(scp)):
                        print(scp[i - 1], scp[i])
                        if scp[i] > 151:
                            print('SCP WORKING', scp)
                            data.iloc[sep_list[i - 1] + scp[i - 1]:sep_list[i - 1] + scp[i], :].to_csv(
                                save_path + 'OrderRight_' + name.strip('.csv') + '_' + str(i).zfill(
                                    4) + '_scpindex_' + str(sep_list[i - 1] + scp[i]) + '.csv')

                        else:
                            print('seq is less length at starting')
                            data.iloc[sep_list[i - 1] + scp[i - 1]:sep_list[i - 1] + scp[i], :].to_csv(
                                save_path + 'LessLength_' + name.strip('.csv') + '_' + str(i).zfill(
                                    4) + '_scpindex_' + str(sep_list[i - 1] + scp[i]) + '.csv')

            else:
                if Less_Than(list(seq)):
                    data.iloc[sep_list[i - 1]:sep_list[i], :].to_csv(
                        save_path + 'LessLength' + name.strip('.csv') + '_' + str(i).zfill(4) + '.csv')
                    print('LessLength!! {} '.format(i))
                else:
                    data.iloc[sep_list[i - 1]:sep_list[i], :].to_csv(
                        save_path + 'LessLength_' + 'OrderError_' + name.strip('.csv') + '_' + str(i).zfill(4) + '.csv')
                    print('OrderError!! LessLength!! for {} '.format(i) * 8)


def main():
    """
    :return:
    """
    # data load
    data_path = '/Users/xingqiangchen/Desktop/2019-02-22/data/'
    data_save_path = '/Users/xingqiangchen/Desktop/2019-02-22/data_check/'

    if not os.path.exists(data_save_path):
        os.makedirs(data_save_path)

    data_name = ['10.csv', '56.csv', '64.csv', '72.csv', '84.csv',
                 '11.csv', '56_0.csv', '64_0.csv', '73.csv', '85.csv',
                 '12.csv', '57.csv', '65.csv', '74.csv', '86.csv',
                 '40.csv', '60.csv', '66.csv', '77.csv', '87.csv',
                 '45.csv', '60_0.csv', '67.csv', '78.csv', '89.csv',
                 '50.csv', '61.csv', '68.csv', '80.csv',
                 '51.csv', '62.csv', '69.csv', '81.csv',
                 '52.csv', '63.csv', '70.csv', '82.csv',
                 '55.csv', '63_2.csv', '71.csv', '83.csv']

    pre_process(data_name, data_path, data_save_path)


if __name__ == '__main__':
    main()
