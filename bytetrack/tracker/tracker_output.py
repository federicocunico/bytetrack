from typing import List
import numpy as np
from pydantic import BaseModel


class TrackerOutput(BaseModel):
    track_id: int
    tlwh: List[int]
    score: float

    # @property
    def tlwh_orig_frame(self):
        return self.tlwh
