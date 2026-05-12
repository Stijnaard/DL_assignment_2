# imports:
#//>>
from dl_assignment_2.data.dataSegment import DataSegment
from typing import List
#//<<

class SubjectData:
    """
    This class couples multiple DataSegments together under a single test subject.
    While I have written this, and other classes in such a way that the manual instantiation of this class will not be necessary,
    it absolutely is possible. In fact, there are multiple ways to achieve the same result as shown below:
    --> some_SubjectData: SubjectData = SubjectData(id=123456, data_file_paths=["path1.h5", ..., "path_n.h5"])

    orrr..
    --> some_SubjectData: SubjectData(id=123456, []) # yes, instantiate with no data
    --> some_SubjectData.add_segment(segment_belonging_to_subject)

    Attributes:
        data_segments: List of DataSegment objects related to one particular subject (id).
    """
    data_segments: List[DataSegment] = []
    def __init__(self, subject_id: int, data_file_paths: List[str] = []) -> None:
        #//>>
        self.subject_id: int = subject_id
        self.data_segments: List[DataSegment] = [DataSegment(path) for path in data_file_paths]

        self._check_data_ownership()
        return None
        #//<<

    def get_subject_id(self) -> int:
        #//>>
        return self.subject_id
        #//<<

    def _check_data_ownership(self) -> None:
        #//>>
        for dataSegment in self.data_segments:
            if dataSegment.get_subject_id() != self.subject_id:
                raise ValueError(f"data segment for: {dataSegment.get_subject_id()} does not belong to {self.subject_id} SubjectData")

        return None
        #//<<

    def get_data_for_task(self, task: str) -> List[DataSegment]:
        #//>>
        """returns only the DataSegments that are related to a particular task:

        Args:
            task: string representation of the task at hand.

        Returns:
            list of DataSegments whose task matches the one given in the input.
        """
        task_specific_segments: List[DataSegment] = []

        for segment in self.data_segments:
            if segment.get_task() == task:
                task_specific_segments.append(segment)

        return task_specific_segments
        #//<<

    def add_segment(self, segment: DataSegment) -> None:
        #//>>
        self._check_segment_ownership(segment)

        self.data_segments.append(segment)
        #//<<

    def _check_segment_ownership(self, dataSegment: DataSegment) -> None:
        #//>>
        if self.subject_id != dataSegment.get_subject_id():
            ValueError(f"data segment for: {dataSegment.get_subject_id()} does not belong to {self.subject_id} SubjectData")

        return None
        #//<<
