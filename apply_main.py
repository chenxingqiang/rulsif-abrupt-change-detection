from src.driving_data_preprocess import apply_preprocess
from src.parallel_aggressive_driving_detection import apply_detection
from src.find_aggressive_driving_event import find_event


def main():
    apply_preprocess()
    apply_detection()
    find_event()


if __name__ == '__main__':
    main()
