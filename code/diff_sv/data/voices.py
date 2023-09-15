import os
from dataclasses import dataclass

@dataclass
class TestItem:
    key: str
    path: str

@dataclass
class TestTrial:
    key1: str
    key2: str
    label: str

class Voices:
    @property
    def dev_set(self):
        return self.__dev_set

    @property
    def eval_set(self):
        return self.__eval_set

    @property
    def dev_trials(self):
        return self.__dev_trials

    @property
    def eval_trials(self):
        return self.__eval_trials

    def __init__(self, path_dev, path_dev_trial, path_eval, path_eval_trial):
        
        # dev_set
        self.__dev_set = []
        self._parse_item(path_dev, self.__dev_set)

        # dev_set
        self.__eval_set = []
        self._parse_item(path_eval, self.__eval_set)
        
        # test_trials
        self.__dev_trials = self._parse_trials(path_dev_trial)
        
        # test_trials
        self.__eval_trials = self._parse_trials(path_eval_trial)
        
    def _parse_item(self, path, dataset):    
        for root, _, files in os.walk(path):
            for file in files:
                # if '.wav' in file:
                if file[-4:] == '.wav'  and file[:2] != '._':
                    temp = os.path.join(root, file)
                    dataset.append(
                        TestItem(
                            path=temp,
                            key=file[:-4]
                        )
                    )

    def _parse_trials(self, path):
        trials = []

        f = open(path) 
        for line in f.readlines():
            strI = line.split(' ')
            trials.append(
                TestTrial(
                    key1=strI[0].replace('\n', ''),
                    key2=strI[1].split('/')[-1].split('.')[0].replace('\n', ''),
                    label='0' if strI[-1].strip() == 'imp' else '1'
                )
            )
        
        return trials