#imports:
#//>>
from typing import List
from os.path import abspath
#//<<

"""
This config contains all the variables that get used in multiple files.
Generally, the variables depicted here show the relative paths to certain folders containing data.
files like dataFolderReader.py rely on the paths depicted by these variables.
"""

# paths to the folders for the cross-training files:

CROSS_TEST_FOLDERS: List[str] = ["test1", "test2", "test3"]

CROSS_ROOT: str = "dl_assignment_2/datasets/Cross"
CROSS_TRAIN: str = "datasets/Cross/train"
CROSS_TEST: str = "datasets/Cross/test"

# paths to the folders for the intra-training files:
INTRA_ROOT: str = "datasets/Intra"
INTRA_TRAIN: str = 'datasets/Intra/train'
INTRA_TEST: str = 'datasets/Intra/train'

TASK_TYPES: List[str] = ["rest", "task_motor", "task_story_math", "task_working_memory"]
