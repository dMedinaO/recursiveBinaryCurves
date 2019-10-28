class checkKValue(object):

    def __init__(self, dataSet):

        self.dataSet = dataSet

        if len(self.dataSet)>=200:
            self.kvalue=10
        elif len(self.dataSet)<200 and len(self.dataSet)>=100:
            self.kvalue=5
        else:
            self.kvalue=-1#implica que se utilizara validacion cruzada con Leave One Out
