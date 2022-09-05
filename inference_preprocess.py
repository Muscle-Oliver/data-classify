# coding: UTF-8
from typing import Any, Optional, Callable
from clearml import Task
#import torch
import json
import numpy as np


# Notice Preprocess class Must be named "Preprocess"
class Preprocess(object):
    def __init__(self):
        # set internal state, this will be called only once. (i.e. not per request)
        task = Task.get_task(project_name='HPO Auto-training',
                             task_name='Datatype Classifier',
                             task_filter={'status': ['completed']})
        word2index_path = task.artifacts['word2index'].get_local_copy()
        self.word2index = json.loads(open(word2index_path, "r").read())
        labels2index_path = task.artifacts['labels2index'].get_local_copy()
        labels2index = json.loads(open(labels2index_path, "r").read())
        self.index2labels = dict(zip(labels2index.values(), labels2index.keys()))
        inference_info_path = task.artifacts['Inference info'].get_local_copy()
        self.inference_info = json.loads(open(inference_info_path, "r").read())
        '''
        acc = np.array([float(x.config_text.split(': ')[-1]) for x in task.models['output']])
        best_index = np.argmax(acc, axis=0)
        self.model_path = task.models['output'][best_index].get_local_copy()
        self._model = None
        '''

    '''
    def load(self, local_file_name: str) -> Optional[Any]:
        self._model = torch.jit.load(self.model_path)
    '''
    
    def pad(self, x, max_len):
        PAD ='<PAD>'
        x += [self.word2index.get(PAD)] * max_len
        return x[:max_len]

    def preprocess(self, body: dict, state: dict, collect_custom_statistics_fn=None) -> Any:
        UNK = '<UNK>'
        text = body.get("text")
        if not text:
            raise ValueError("'text' entry not provided, expected a json dict with key 'text.'")

        max_len = self.inference_info['max_input_length']
        sentence = self.pad([self.word2index.get(word, self.word2index.get(UNK)) for word in text], max_len)
        return sentence

    '''
    def process(self, data: Any, state: dict, collect_custom_statistics_fn: Optional[Callable[[dict], None]]) -> Any:
        data = self._model(data)
        return data
    '''
    
    def postprocess(self, data: Any, state: dict, collect_custom_statistics_fn=None) -> dict:
        # post process the data returned from the model inference engine
        # data is the return value from model.predict we will put is inside a return value as Y
        if not isinstance(data, np.ndarray):
            # this should not happen
            return dict(result=-1)
        result = dict(result=self.index2labels[np.argmax(data)])
        return result

if __name__ == "__main__":
    pass
