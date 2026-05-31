from dl_assignment_2.data.dataSegment import DataSegment
from dl_assignment_2.data.segmentHelpers import SegmentInfo

from typing import Optional, Callable

from matplotlib.axes import Axes
import matplotlib.pyplot as plt


class ResidualSegment(DataSegment):
    def plot(self, axis: Optional[Axes] = None) -> Axes:
        axis_given: bool = True
        
        if not axis:
            axis_given = False
            _, axis = plt.subplots()
            
        im = axis.imshow(self.data, aspect='auto')
        axis.set_title(f"subject: {self.subject_id}'s {self.segment}-residual segment\n for {self.task}")
        plt.colorbar(im, ax=axis)
        
        if not axis_given:
            plt.show()
            
        return axis
    
    def plot_element_distribution(self, n_bins: int, axis: Optional[Axes]) -> Axes:
        axis_given: bool = True
        
        if not axis:
            axis_given = False
            _, axis = plt.subplots()
            
            
        im = axis.hist(self.data.flatten(), n_bins)
        axis.set_title(f"subject: {self.subject_id}'s residual element distribution for\n{self.task}[{self.segment}]")
        
        if not axis_given:
            plt.show()
            
        return axis
    
    def transform(self, transformation_function: Callable) -> "ResidualSegment":
        data = transformation_function(self.data)

        return ResidualSegment(info=SegmentInfo(data=data, 
                                subject_id=self.subject_id, 
                                task=self.task, 
                                segment=self.segment))
    