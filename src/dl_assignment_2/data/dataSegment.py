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
        print(f"given path: {path}")
        
        task, subject_id, segment = path.split("_")
        
        segment, _ = segment.split(".")
        task = task.split("/")[-1]
        
        converted_subject_id: int = int(subject_id)
        converted_segment: int = int(segment)
        return task, converted_subject_id, converted_segment

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