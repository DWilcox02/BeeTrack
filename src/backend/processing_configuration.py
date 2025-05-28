
class ProcessingConfiguration():
    def __init__(
        self, 
        smoothing_alpha: float,
        dbscan_epsilon: float,
        deformity_delta: float,
        processing_seconds: int
    ):
        self.smoothing_alpha = smoothing_alpha
        self.dbscan_epsilon = dbscan_epsilon
        self.deformity_delta = deformity_delta
        self.processing_seconds = processing_seconds