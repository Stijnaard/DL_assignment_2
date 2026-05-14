# imports:
#//>>
import h5py
from h5py import File
from numpy import ndarray, array
import matplotlib.pyplot as plt

from typing import Tuple

from dl_assignment_2.data.data_config import INTRA_TRAIN
#//<<

class DataSegment:
    """
    The DataSegment class aims to represent a single file of data.
    Its existence aims to streamline the process of opening the file and converting its contents to numpy's ndarray.
    It is advices to use the variables set in the data.config file for easily accessing the relevant folders

    How to use it:
    -> some_dataSegment: DataSegment = DataSegment(path="path/to/file.h5")

    Attributes:
        data: ndarray version of the data present in the given file.
        subject_id: id of the subject who the data belongs to.
        task: task the subject was performing at the time the neural activity was being measured.
        segment: integer corresponding to final integer in the file name. It is a segment/snippet of the entire recording session.
    """
    data: ndarray
    subject_id: int
    task: str
    segment: int

    def __init__(self, relative_path: str) -> None:
        #//>>
        task, subject_id, segment = self._get_metadata(relative_path)
        self.task: str = task
        self.subject_id: int = subject_id
        self.segment: int = segment

        self.data: ndarray = self._read_segment_from_file(relative_path)
        
        return None
        #//<<

    def __repr__(self) -> str:
        return f"data segment concerning subject: {self.subject_id} for task: {self.task} at segment: {self.segment}."
    def _read_segment_from_file(self, path: str) -> ndarray:
        #//>>
        file: File = h5py.File(path, 'r')
        key_name: str = f"{self.task}_{self.subject_id}"

        data: ndarray = array(file[key_name])

        return data
        #//<<

    @staticmethod
    def _get_metadata(path: str) -> Tuple[str, int, int]:
        #//>>
        file_name_without_extension: str = DataSegment._get_filename_without_extension(path)

        task: str = DataSegment._get_task_from_fileName(file_name_without_extension)
        segment = file_name_without_extension.split("_")[-1]
        subject_id: int = file_name_without_extension.split("_")[-2]

        converted_subject_id: int = int(subject_id)
        converted_segment: int = int(segment)

        return task, converted_subject_id, converted_segment
        #//<<

    @staticmethod
    def _get_task_from_fileName(file_name: str) -> str:
        #//>>
        if file_name.startswith("rest"):
            return "rest"
        elif file_name.startswith("task_motor"):
            return "task_motor"
        elif file_name.startswith("task_story_math"):
            return "task_story_math"
        else:
            return "task_working_memory"
        #//<<
        
    @staticmethod
    def _get_filename_without_extension(path: str) -> str:
        #//>>
        file_name: str =path.split("/")[-1]
        file_name_without_extension: str = file_name.replace(".h5", "")
        return file_name_without_extension
        #//<<

    # all the getters:
    def get_data(self) -> ndarray:
        #//>>
        return self.data
        #//<<
    
    def get_subject_id(self) -> int:
        #//>>
        return self.subject_id
        #//<<
    
    def get_task(self) -> str:
        #//>>
        return self.task
        #//<<
    
    def get_segment(self) -> int:
        #//>>
        return self.segment
        #//<<

    def plot(self) -> None:
        plt.imshow(self.data, aspect='auto')
        plt.show()
        return None

