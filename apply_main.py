from src.config import first_time
from src.driving_data_preprocess import apply_preprocess
from src.find_aggressive_driving_event import find_event
from src.parallel_aggressive_driving_detection import apply_detection


def main():

    if first_time:
        apply_preprocess()
    else:
        pass

    apply_detection()
    find_event()


if __name__ == '__main__':
    main()
