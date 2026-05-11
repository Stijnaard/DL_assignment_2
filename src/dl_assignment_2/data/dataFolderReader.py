from typing import List

class DataFolderReader:
    tasks: List[str] = []
    subject_IDs: List[int] = []
    
    
    
    def __init__(self, root: str, subject_id: int) -> None:
        self.root: str = root
        self.subject_id: int = subject_id
        
        

        
        