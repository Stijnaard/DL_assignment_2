from dl_assignment_2.data.absPathProvider import AbsPathProvider
from dl_assignment_2.data.data_config import TASK_TYPES
from dl_assignment_2.data.dataSegment import DataSegment, SegmentSummary

from typing import Callable
from os import scandir

import matplotlib.pyplot as plt

class BetweenTaskAnalysis:
    """This class concerns itself with streamlining the analysis of Datasegments.
    The scope of this class is comparing different DataSegments from the same subject and segment, but for different tasks"""
    TASK_TYPES: list[str] = TASK_TYPES
    abp: AbsPathProvider = AbsPathProvider()
    
    def __init__(self, folder_path: str = AbsPathProvider().get_intra_train_path(), wanted_segment: int = 1) -> None:
        self.task_to_segment: dict[str, DataSegment] = {}
        self.folder_path: str = folder_path
        self.segment: int = wanted_segment
        
        for file in scandir(folder_path):
            task: str
            segment: int # type: ignore[assignment]
            task, _, segment = DataSegment._get_metadata_from_filename(file.name) # type: ignore[assignment]
            
            if segment == wanted_segment:
                abs_file_path: str = self.abp.get_abs_path_to_segment_file(self.abp.get_intra_train_path(), file.name)
                segment: DataSegment = DataSegment(abs_file_path)
                self.task_to_segment[task] = segment
                
        return None
    
    def transform(self, trans_func: Callable = lambda x: x*10**12) -> None:
        """Applies transformation function to all segments it has"""
        for task in self.task_to_segment:
            old_segment: DataSegment = self.task_to_segment[task]
            
            transformed_segment: DataSegment = old_segment.transform(trans_func)
            
            self.task_to_segment[task] = transformed_segment
            
    def plot(self, way: str = "matrix", n_bins: int = 15) -> None:
        """Plots all segments at once"""
        _, axis = plt.subplots(2,2)
        
        for idx, (task, segment) in enumerate(self.task_to_segment.items()):
            row_idx: int = idx//2
            column_idx: int = idx%2
            if way == "matrix":
                axis[row_idx, column_idx] = segment.plot(axis=axis[row_idx, column_idx]) 
            elif way == "hist":
                axis[row_idx, column_idx] = segment.plot_element_distribution(n_bins, axis[row_idx,column_idx])
            elif way == "lines":
                axis[row_idx, column_idx] = segment.plot_as_lines(axis=axis[row_idx, column_idx])
        plt.show()
    
    def summarize(self) -> None:
        """Summarizes all segments:"""
        summarizations: list[SegmentSummary] = [segment.summarize() for segment in self.task_to_segment.values()]
        
        for summary in summarizations:
            print(f"{summary.task:<20} w. mean: {summary.mean}, max: {summary.max}, min: {summary.min}")
    
    def copy(self) -> "BetweenTaskAnalysis":
        bta: BetweenTaskAnalysis = BetweenTaskAnalysis(self.folder_path, self.segment)
        return bta
    
    def get_residuals(self) -> None:
        for task in self.TASK_TYPES:
            residual_segment = self.task_to_segment[task].get_residuals()
            self.task_to_segment[task] = residual_segment

    def trim(self, step: int = 8) -> None:
        for task in self.TASK_TYPES:
            trimmed_segment = self.task_to_segment[task].trim(step)
            self.task_to_segment[task] = trimmed_segment
            
        return None


if __name__ == "__main__":
    bta = BetweenTaskAnalysis()
    bta.trim(100)
    bta.transform()
    bta.summarize()
    #bta.plot()
    #bta.plot(way="hist")
    bta.plot(way="lines")
    print("\n\n")
    
    bta.get_residuals()
    bta.transform(lambda x: x*10**3)
    bta.summarize()
    bta.plot()
    bta.plot(way="hist")
    
    print(type(bta.task_to_segment["rest"]))
    