import os
import sys
import json
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'langeval'))


class LangEvaluator():
    def __init__(self, dataset):
        self.uid2ref = {}
        for datum in dataset.data:
            self.uid2ref[datum['uid']] = datum['api'] 
            
 