class ResultsLoggingService:
    """Stores the results of the quantization attempts and the best quantization level."""
    result_store : List[dict[str, str | None]]
    row_number    : int
    
    def __init__(self):
        self.result_store = {}
        self.row_number = 0
    
    def record_result(self, result: dict[str, str | None]):
        self.result_store.append(result)
        self.row_number += 1
    
    def print_results(self):
        for result in self.result_store:
            print(result)