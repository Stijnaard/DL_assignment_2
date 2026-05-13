# Imports:
#//>>
from dl_assignment_2.data.data_config import INTRA_TRAIN, INTRA_TEST, CROSS_TRAIN, CROSS_TEST

from os.path import abspath
#//<<

class AbsPathProvider:
    ABS_PART: str = abspath(".").split("src")[0] # before the src part, stuff can vary, so we need that part only

    INTRA_TRAIN: str = f"{ABS_PART}/{INTRA_TRAIN}"
    INTRA_TEST: str = f"{ABS_PART}/{INTRA_TEST}"

    CROSS_TRAIN: str = f"{ABS_PART}/{CROSS_TRAIN}"
    CROSS_TEST: str = f"{ABS_PART}/{CROSS_TEST}"

    def get_intra_train_path(self) -> str:
        return self.INTRA_TRAIN

    def get_intra_test_path(self) -> str:
        return self.INTRA_TEST


    def get_cross_train_path(self) -> str:
        return self.CROSS_TRAIN

    def get_cross_test_path(self) -> str:
        return self.CROSS_TEST