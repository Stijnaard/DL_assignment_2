"""
This file contains the class that is responsible for performing analysis on individual segments. 
The main goal was just plotting the segments and maybe trying some transformations of the data.

However, the analysis has more so shifted from studying individual segments to studying different segments belonging to different tasks.
To keep these two types of analyis seperate, some refactoring would have to get performed.
"""

# imports:
#//>>
from dl_assignment_2.data.dataSegment import DataSegment
from dl_assignment_2.data.dataFolderReader import DataFolderReader
from dl_assignment_2.data.subjectData import SubjectData
from dl_assignment_2.data.absPathProvider import AbsPathProvider
from dl_assignment_2.data.data_config import TASK_TYPES

from numpy import ndarray, array
import matplotlib.pyplot as plt

from typing import List, Dict
#//<<

class SegmentAnalysis:
    pathProvider: AbsPathProvider = AbsPathProvider()
    intra_training_data: DataFolderReader = DataFolderReader(pathProvider.get_intra_train_path())

    task_types: List[str] = TASK_TYPES
    x: SubjectData = intra_training_data[0]

    task_to_data_map: Dict[str, ndarray] = {}
    for task in task_types:
        task_specific_segment: DataSegment = x.get_data_for_task(task)[0]
        task_to_data_map[task] = task_specific_segment.get_data()

    def show_segment_for_each_task(self) -> None:

        _, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

        # Map tasks to titles and subplot positions
        plot_config = [
            (axes[0, 0], "rest", "rest segment"),
            (axes[0, 1], "task_motor", "motoric segment"),
            (axes[1, 0], "task_story_math", "math & story segment"),
            (axes[1, 1], "task_working_memory", "working memory segment"),
        ]

        for ax, task_key, title in plot_config:
            im = ax.imshow(self.task_to_data_map[task_key], aspect="auto")
            ax.set_title(title)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.show()

    def check_negative_value_presence(self) -> None:
       raise NotImplementedError("still gotta implement this") 
    
    def plot_data_as_lines(self) -> None:
        _, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        _, line_summary_axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

        # Map tasks to titles and subplot positions
        plot_config = [
            (axes[0, 0], "rest", "rest segment"),
            (axes[0, 1], "task_motor", "motoric segment"),
            (axes[1, 0], "task_story_math", "math & story segment"),
            (axes[1, 1], "task_working_memory", "working memory segment"),
        ]

        for ax, task_name, title in plot_config:
            task_data: ndarray = self.task_to_data_map[task_name]*10**10
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
    
    def plot_means_hist(self) -> None:
        _, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

        # Map tasks to titles and subplot positions
        plot_config = [
            (axes[0, 0], "rest", "rest segment"),
            (axes[0, 1], "task_motor", "motoric segment"),
            (axes[1, 0], "task_story_math", "math & story segment"),
            (axes[1, 1], "task_working_memory", "working memory segment"),
        ]

        for ax, task_name, title in plot_config:
            task_data: ndarray = self.task_to_data_map[task_name]*10**10
            line_means: List[float] = []
            for row_idx in range(0, task_data.shape[0]-200):
                row: ndarray = task_data[row_idx,1000:2500].flatten()
                row_avg: float = row.mean()

                line_means.append(row_avg)

            ax.hist(line_means, bins=10, edgecolor="black")
            ax.set_title(title)

        plt.show()

    def plot_std_dev_hist(self) -> None:
        _, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

        # Map tasks to titles and subplot positions
        plot_config = [
            (axes[0, 0], "rest", "rest segment"),
            (axes[0, 1], "task_motor", "motoric segment"),
            (axes[1, 0], "task_story_math", "math & story segment"),
            (axes[1, 1], "task_working_memory", "working memory segment"),
        ]

        for ax, task_name, title in plot_config:
            task_data: ndarray = self.task_to_data_map[task_name]*10**10

            line_std_devs: List[float] = []
            for row_idx in range(0, task_data.shape[0]-200):
                row: ndarray = task_data[row_idx,1000:2500].flatten()
                row_std_dev: float = row.std()

                line_std_devs.append(row_std_dev)

            ax.hist(line_std_devs, bins=10, edgecolor="black")
            ax.set_title(title)

        plt.show()