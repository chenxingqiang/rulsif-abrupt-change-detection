# -*- coding: utf-8 -*-
# @Create Time    : 2019-03-27 12:33
# @Author  : Xingqiang Chen
# @Software: PyCharm

from src.config import first_time, only_evaluation
from src.driving_data_preprocess import apply_preprocess
from src.find_aggressive_driving_event import find_event
from src.parallel_aggressive_driving_detection import apply_detection


def main():

    if first_time:
        print("!! RUNNING FOR FIRST TIME, DO DATA PREPROCESS>>>>>")
        apply_preprocess()
    else:
        print("!! SKIP DATA PREPROCESS>>>>>> ")
        pass
    if not only_evaluation:
        print("!! RUNNING FOR ABRUPT-CHANGE DETECTION>>>>>>>>")
        apply_detection()
    else:
        print("!! ONLY EVALUATION AND FIND EVENTS STATE>>>>>>>>> ")
        pass

    find_event()


if __name__ == '__main__':
    main()
