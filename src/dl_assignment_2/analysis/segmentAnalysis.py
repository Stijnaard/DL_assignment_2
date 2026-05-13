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

    def show_segment_for_each_task(self) -> None:
        task_to_data_map: Dict[str, ndarray] = {}
        for task in self.task_types:
            task_specific_segment: DataSegment = self.x.get_data_for_task(task)[0]
            task_to_data_map[task] = task_specific_segment.get_data()

        _, ax = plt.subplots(2, 2)

        # this whole part could get refactored:
        ax[0,0].imshow(task_to_data_map["rest"], aspect="auto")
        ax[0,0].set_title("rest segment")

        ax[0,1].imshow(task_to_data_map["task_motor"], aspect="auto")
        ax[0,1].set_title("motoric segment")

        ax[1,0].imshow(task_to_data_map["task_story_math"], aspect="auto")
        ax[1,0].set_title("math & story segment")

        ax[1,1].imshow(task_to_data_map["task_working_memory"], aspect="auto")
        ax[1,1].set_title("working memory segment")

        plt.show()


if __name__ == "__main__":
    x: SegmentAnalysis = SegmentAnalysis()
    x.show_segment_for_each_task()
