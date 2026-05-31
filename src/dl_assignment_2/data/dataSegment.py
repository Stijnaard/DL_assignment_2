# imports:
#//>>
import h5py
from h5py import File
from numpy import ndarray, array, delete, zeros, concatenate
import matplotlib.pyplot as plt
from matplotlib import axes

from typing import Tuple, Callable, Optional
from dataclasses import dataclass
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

    def __init__(self, relative_path: Optional[str] = None, dataSegmentInfo: Optional["DataSegmentInfo"] = None) -> None:
        #//>>
        if relative_path:
            self._construct_segment_from_file(relative_path)
        elif dataSegmentInfo:
            self._construct_segment_from_DataSegmentInfo(dataSegmentInfo)
        else:
            raise ValueError("relative path AND dataSegmentInfo variable have been left unspecified.")
        
        return None
    
    def transform(self, transformation_function: Callable) -> "DataSegment":
        data = transformation_function(self.data)

        return DataSegment(dataSegmentInfo=DataSegmentInfo(data=data, 
                               subject_id=self.subject_id, 
                               task=self.task, 
                               segment=self.segment))

    # all the analytical functions:
    ###############################
    def plot(self) -> None:
        plt.imshow(self.data, aspect='auto')
        plt.show()
        return None
    
    def trim_n_rows(self, n: int = 3) -> "DataSegment":
        "removes all rows but every N one"
        
        kept_indices: list[int] = self._compute_indices_to_keep(n)
        removed_indices: list[int] = self._compute_indices_to_remove(kept_indices)

        trimmed_data: ndarray
        if len(self.data.shape) == 2:
            trimmed_data =  delete(self.data, removed_indices, 1)

        else:
            trimmed_data = delete(self.data, removed_indices) 
            
        return DataSegment(dataSegmentInfo=DataSegmentInfo(trimmed_data, self.subject_id, self.task, self.segment))

        
    def get_residuals(self) -> "DataSegment":
        """Computes matrix where each column i consists of the difference
        between the the columns i and i+1 of the DataSegment's data
        """
        # 1. construct the padding column:
        row_amount: int = self.data.shape[0]
        padding_column: ndarray = zeros(shape=(row_amount, 1))
        
        # 2. construct the right-shifted copy of the data:
        copy_without_final_column: ndarray = self.data.copy()[:,:-1]
        padded_right_shifted_copy: ndarray = concatenate((padding_column, copy_without_final_column), axis=1)
        
        # 3. construct the residuals:
        padded_residuals: ndarray = self.data - padded_right_shifted_copy
        
        # 4. remove the padded first column
        unpadded_residuals: ndarray = padded_residuals[:,1:]
        
        return DataSegment(dataSegmentInfo=DataSegmentInfo(unpadded_residuals, self.subject_id, self.task, self.segment))


    # all the getters:
    ##################
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
    
    # init helpers:
    ###############
    def _construct_segment_from_DataSegmentInfo(self, dataSegmentInfo: "DataSegmentInfo") -> None:
        self.data: ndarray = dataSegmentInfo.data
        self.subject_id: int = dataSegmentInfo.subject_id
        self.task: str = dataSegmentInfo.task
        self.segment: int = dataSegmentInfo.segment
        return None
        
    def _construct_segment_from_file(self, relative_path):
        task, subject_id, segment = self._get_metadata_from_filename(relative_path)
        self.task: str = task
        self.subject_id: int = subject_id
        self.segment: int = segment

        self.data: ndarray = self._read_segment_from_file(relative_path)
        #//<<
        
    def _read_segment_from_file(self, path: str) -> ndarray:
        #//>>
        file: File = h5py.File(path, 'r')
        key_name: str = f"{self.task}_{self.subject_id}"

        data: ndarray = array(file[key_name])

        return data
        #//<<
        
    def __repr__(self) -> str:
        return f"data segment concerning subject: {self.subject_id} for task: {self.task} at segment: {self.segment}."
    
    @staticmethod
    def _get_metadata_from_filename(path: str) -> Tuple[str, int, int]:
        #//>>
        file_name_without_extension: str = DataSegment._get_filename_without_extension(path)

        task: str = DataSegment._get_task_from_fileName(file_name_without_extension)
        segment = file_name_without_extension.split("_")[-1]
        subject_id: str = file_name_without_extension.split("_")[-2]

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

    
        
    def _compute_indices_to_keep(self, step) -> list[int]:
        start = step-1
        indices_to_keep = [start]

        s = 1
        while indices_to_keep[-1] < self.data.shape[1]:
            indices_to_keep.append(start + step*s)
            s+=1
        indices_to_keep.pop()

        return indices_to_keep

    def _compute_indices_to_remove(self,indices_to_keep):
        return list(set(range(self.data.shape[1])).difference(indices_to_keep))





@dataclass
class DataSegmentInfo:
    """
    This class helps out with transferring info from one datasegment to another
    when trying to create a copy or new object based on the data of another one.

    It should not be created manually ever.
    """
    data: ndarray
    subject_id: int
    task: str
    segment: int
    
