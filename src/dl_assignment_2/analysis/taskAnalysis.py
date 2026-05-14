"""
This file contains the class that is responsible for performing analysis on multiple segments
revolving around the same subject and segment, but peforming different tasks.
Therefore, the type of analysis done here is mainly to compare the data between tasks and
hopefully gain new, interesting insights.
"""

# imports:
#//>>
from dl_assignment_2.data.dataSegment import DataSegment
from dl_assignment_2.data.absPathProvider import AbsPathProvider

from numpy import ndarray
import matplotlib.pyplot as plt

from typing import List, Dict, Optional, Callable
from os import scandir
#//<<

class TaskAnalysis:
    _, __axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    __plot_config = [
        (__axes[0, 0], "rest", "rest segment"),
        (__axes[0, 1], "task_motor", "motoric segment"),
        (__axes[1, 0], "task_story_math", "math & story segment"),
        (__axes[1, 1], "task_working_memory", "working memory segment"),
    ]

    def __init__(self, folder_path: str = AbsPathProvider().get_intra_train_path(), subject_id: Optional[int] = None, segment: int = 1) -> None:
        self.task_to_data_map: Dict[str, ndarray] = {}
        self.segment: int = segment
        self.subject_id: int = subject_id

        for file in scandir(folder_path):
            full_path: str = f"{folder_path}/{file.name}"
            dataSegment: DataSegment = DataSegment(full_path)
            
            if not subject_id:
                subject_id: int = dataSegment.get_subject_id()
                self.subject_id = subject_id

            else:
                if dataSegment.get_segment() == segment:
                    self.task_to_data_map[dataSegment.get_task()] = dataSegment.get_data()

        return None
    
    def __repr__(self) -> str:
        return f"TaskAnalysis object for subject: {self.subject_id} for segment: {self.segment}."
    
    def transform(self, transformation_function: Callable) -> None:
        for name in self.task_to_data_map:
            data: ndarray = self.task_to_data_map[name]

            transformed_data: ndarray = transformation_function(data)

            self.task_to_data_map[name] = transformed_data

        return None

    def plot_segment_per_task(self) -> None:
        for ax, task_key, title in self.__plot_config:
            im = ax.imshow(self.task_to_data_map[task_key], aspect="auto")
            ax.set_title(title)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.show()

    def check_negative_value_presence(self) -> None:
       raise NotImplementedError("still gotta implement this") 
    
    def plot_data_as_lines(self) -> None:
        for ax, task_name, title in self.__plot_config:
            task_data: ndarray = self.task_to_data_map[task_name]
            
            line_means: List[float] = []
            line_std_devs: List[float] = []

            for row_idx in range(0, task_data.shape[0]-200):
                row: ndarray = task_data[row_idx,1000:2500].flatten()
                row_avg: float = row.mean()
                row_std_dev: float = row.std()

                line_means.append(row_avg)
                line_std_devs.append(row_std_dev)

                ax.plot(row)
            ax.set_title(title)

        plt.show()

        return None
    
    def plot_means_hist(self) -> None:
        for ax, task_name, title in self.__plot_config:
            task_data: ndarray = self.task_to_data_map[task_name]*10**10
            line_means: List[float] = []
            for row_idx in range(0, task_data.shape[0]-200):
                row: ndarray = task_data[row_idx,1000:2500].flatten()

                row_avg: float = row.mean()

                line_means.append(row_avg)

            ax.hist(line_means, bins=10, edgecolor="black")
            ax.set_title(title)

        plt.show()

        return None

    def plot_std_dev_hist(self) -> None:
        for ax, task_name, title in self.__plot_config:
            task_data: ndarray = self.task_to_data_map[task_name]*10**10

            line_std_devs: List[float] = []
            for row_idx in range(0, task_data.shape[0]-200):
                row: ndarray = task_data[row_idx,1000:2500].flatten()

                row_std_dev: float = row.std()

                line_std_devs.append(row_std_dev)

            ax.hist(line_std_devs, bins=10, edgecolor="black")
            ax.set_title(title)

        plt.show()
        
        return None

if __name__ == "__main__":
    ta: TaskAnalysis = TaskAnalysis()
    ta.plot_segment_per_task()
    ta.plot_data_as_lines()
    ta.plot_means_hist()
    ta.plot_std_dev_hist()