from src.config import first_time
from src.driving_data_preprocess import apply_preprocess


def main():

    if first_time:
        apply_preprocess()
    else:
        pass

    # apply_detection()
    # find_event()


if __name__ == '__main__':
    main()
