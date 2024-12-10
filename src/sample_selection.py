from optimizer import OptimizedCodeSchema
class SampleSelection:
    def __init__(self, results: list[OptimizedCodeSchema]):
        self.correct_list = []
        self.incorrect_list = []
        for result in results:
            if result.opt == 1 and result not in self.correct_list:
                self.correct_list.append(result)
            elif self.opt < 1:
                self.incorrect_list.append(result)
        
        

    