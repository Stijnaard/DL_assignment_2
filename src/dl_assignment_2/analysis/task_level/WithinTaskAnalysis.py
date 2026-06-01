from dl_assignment_2.data.absPathProvider import AbsPathProvider
from dl_assignment_2.data.dataSegment import DataSegment, SegmentInfo

from numpy import ndarray, concatenate

from typing import Optional
from os import scandir

class WithinTaskAnalysis:
    """This class only loads the datasegments from a folder that correspond to the given task and subject_id.
    In contrast to the BetweenTaskAnalysis class, this class only centers around the segments for a single task and single person,
    and hopes to find patterns that are unique to that specific task."""
    
    app: AbsPathProvider = AbsPathProvider()
    segments: list[DataSegment]
    full_segment: DataSegment
    subject: int
    task: str

    def __init__(self, abs_folder_path: Optional[str] = None, segments: list[DataSegment] = None, task: Optional[str] = None, subject: Optional[int] = None) -> None:
        # 0. init everything:
        self.segments: list[DataSegment] = []
        self.subject = subject
        self.task = task

        # 1. construct the segments:
        if abs_folder_path:
            self.segments = self._construct_segments_from_folder(abs_folder_path, task, subject)
        elif segments:
            self.segments = segments
        else:
            raise ValueError("abs_folder_path or segments must be given")

        # 2. sort the segments in ascending order on segment:
        self.segments = sorted(self.segments, key=lambda x: x.get_segment())

        # 3. construct the full segment:
        self._construct_full_segment()

        return None

    def _construct_segments_from_folder(self, abs_folder_path, task, subject) -> list[DataSegment]:
        segments: list[DataSegment] = []
        for file in scandir(abs_folder_path):
            # 2. get the metadata:
            file_task, file_subject, _ = DataSegment._get_metadata_from_filename(file.name)

            # 3. if it belongs, load and add it:
            if file_task == task and file_subject == subject:
                abs_file_path: str = self.app.get_abs_path_to_segment_file(abs_folder_path, file.name)
                ds: DataSegment = DataSegment(abs_file_path)
                segments.append(ds)

        return segments

            
    
    def _construct_full_segment(self) -> None:
        """constructs the full segment where the data from all DataSegments get concatenated column-wise."""
        # 1. get data from all segments:
        all_data: list[ndarray] = [segment.data for segment in self.segments]

        # 2. construct concatenated data:
        concatenated_data: ndarray = concatenate(all_data, axis=1)
        
        # 3. construct the full segment:
        self.full_segment: DataSegment = DataSegment(info=SegmentInfo(concatenated_data, self.subject, self.task, -1))

        return None

    
    


