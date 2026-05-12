import h5py
from h5py import File
from numpy import ndarray, array

from importlib import resources
from typing import Tuple

from dl_assignment_2.data.data_config import INTRA_TRAIN



class DataSegment:
    data: ndarray
    subject_id: int
    task: str
    segment: int
    
    def __init__(self, relative_path: str) -> None:
        task, subject_id, segment = self._get_metadata(relative_path)
        self.task = task
        self.subject_id = subject_id
        self.segment = segment
        
        self.data = self._read_segment_from_file(relative_path)


    def _read_segment_from_file(self, path: str):
        file: File = h5py.File(path, 'r')
        key_name: str = f"{self.task}_{self.subject_id}"
        
        data: ndarray = array(file[key_name])
        
        return data
    
    @staticmethod
    def _get_metadata(path: str) -> Tuple[str, int, int]:
        file_name_without_extension: str = DataSegment._get_filename_without_extension(path)
        
        task: str = DataSegment._get_task_from_fileName(file_name_without_extension)
        segment = file_name_without_extension.split("_")[-1]
        subject_id: int = file_name_without_extension.split("_")[-2]
        
        converted_subject_id: int = int(subject_id)
        converted_segment: int = int(segment)

        return task, converted_subject_id, converted_segment
    
    @staticmethod
    def _get_task_from_fileName(file_name: str) -> str:
        if file_name.startswith("rest"):
            return "rest"
        elif file_name.startswith("task_motor"):
            return "task_motor"
        elif file_name.startswith("task_story_math"):
            return "task_story_math"
        else:
            return "task_working_memory"
        
    @staticmethod
    def _get_filename_without_extension(path: str) -> str:
        file_name: str =path.split("/")[-1]
        file_name_without_extension: str = file_name.replace(".h5", "")
        return file_name_without_extension

    # all the getters:
    def get_data(self) -> ndarray:
        return self.data
    
    def get_subject_id(self) -> int:
        return self.subject_id
    
    def get_task(self) -> str:
        return self.task
    
    def get_segment(self) -> int:
        return self.segment
    
            
if __name__ == "__main__":
    path: str = f"{INTRA_TRAIN}/rest_105923_1.h5"
    x = DataSegment(path)