import os
from dataclasses import dataclass

NUM_TRAIN_ITEM = 148642
NUM_TRAIN_SPK = 1211
NUM_TRIALS = 37611
NUM_TEST_ITEM = 4874

@dataclass
class TrainItem:
    path: str
    speaker: str
    label: int

@dataclass
class TestItem:
    key: str
    path: str

@dataclass
class TestTrial:
    key1: str
    key2: str
    label: str

class VoxCeleb1:
    def __init__(self, path_train, path_test, path_test_noise, path_trial):

        self.train_set = []

        # train_set
        labels = {}
        num_utt = [0 for _ in range(NUM_TRAIN_SPK)]
        num_sample = 0
        for root, _, files in os.walk(path_train):
            for file in files:
                if '.wav' in file:
                    # combine path
                    f = os.path.join(root, file)
                    
                    # parse speaker
                    spk = f.split('/')[-3]
                    
                    # labeling
                    try: labels[spk]
                    except: 
                        labels[spk] = len(labels.keys())

                    # init item
                    item = TrainItem(path=f, speaker=spk, label=labels[spk])
                    self.train_set.append(item)
                    num_sample += 1
                    num_utt[labels[spk]] += 1         

        # test_set
        self.test_set = {
            'clean': [],
            'noise_0': [],
            'noise_5': [],
            'noise_10': [],
            'noise_15': [],
            'noise_20': [],
            'speech_0': [],
            'speech_5': [],
            'speech_10': [],
            'speech_15': [],
            'speech_20': [],
            'music_0': [],
            'music_5': [],
            'music_10': [],
            'music_15': [],
            'music_20': [],
            'nonspeech_0': [],
            'nonspeech_5': [],
            'nonspeech_10': [],
            'nonspeech_15': [],
            'nonspeech_20': []
        }
        self._parse_item(path_test, 'clean')
        self._parse_item(f'{path_test_noise}/noise_0', 'noise_0')
        self._parse_item(f'{path_test_noise}/noise_5', 'noise_5')
        self._parse_item(f'{path_test_noise}/noise_10', 'noise_10')
        self._parse_item(f'{path_test_noise}/noise_15', 'noise_15')
        self._parse_item(f'{path_test_noise}/noise_20', 'noise_20')

        self._parse_item(f'{path_test_noise}/speech_0', 'speech_0')
        self._parse_item(f'{path_test_noise}/speech_5', 'speech_5')
        self._parse_item(f'{path_test_noise}/speech_10', 'speech_10')
        self._parse_item(f'{path_test_noise}/speech_15', 'speech_15')
        self._parse_item(f'{path_test_noise}/speech_20', 'speech_20')

        self._parse_item(f'{path_test_noise}/music_0', 'music_0')
        self._parse_item(f'{path_test_noise}/music_5', 'music_5')
        self._parse_item(f'{path_test_noise}/music_10', 'music_10')
        self._parse_item(f'{path_test_noise}/music_15', 'music_15')
        self._parse_item(f'{path_test_noise}/music_20', 'music_20')

        self._parse_item(f'{path_test_noise}/nonspeech_0', 'nonspeech_0')
        self._parse_item(f'{path_test_noise}/nonspeech_5', 'nonspeech_5')
        self._parse_item(f'{path_test_noise}/nonspeech_10', 'nonspeech_10')
        self._parse_item(f'{path_test_noise}/nonspeech_15', 'nonspeech_15')
        self._parse_item(f'{path_test_noise}/nonspeech_20', 'nonspeech_20')

        # test_trials
        self.test_trials = self._parse_trials(path_trial)
        
        # error check
        assert len(self.train_set) == 148642, f'len(train_set): {len(self.train_set)}'
        assert len(self.test_set["clean"]) == 4874, f'len(test_set): {len(self.test_set["clean"])}'
        assert len(self.test_set["noise_0"]) == 4874, f'len(test_set): {len(self.test_set["noise_0"])}'
        assert len(self.test_set["noise_5"]) == 4874, f'len(test_set): {len(self.test_set["noise_5"])}'
        assert len(self.test_set["noise_10"]) == 4874, f'len(test_set): {len(self.test_set["noise_10"])}'
        assert len(self.test_set["noise_15"]) == 4874, f'len(test_set): {len(self.test_set["noise_15"])}'
        assert len(self.test_set["noise_20"]) == 4874, f'len(test_set): {len(self.test_set["noise_20"])}'
        assert len(self.test_set["speech_0"]) == 4874, f'len(test_set): {len(self.test_set["speech_0"])}'
        assert len(self.test_set["speech_5"]) == 4874, f'len(test_set): {len(self.test_set["speech_5"])}'
        assert len(self.test_set["speech_10"]) == 4874, f'len(test_set): {len(self.test_set["speech_10"])}'
        assert len(self.test_set["speech_15"]) == 4874, f'len(test_set): {len(self.test_set["speech_15"])}'
        assert len(self.test_set["speech_20"]) == 4874, f'len(test_set): {len(self.test_set["speech_20"])}'
        assert len(self.test_set["music_0"]) == 4874, f'len(test_set): {len(self.test_set["music_0"])}'
        assert len(self.test_set["music_5"]) == 4874, f'len(test_set): {len(self.test_set["music_5"])}'
        assert len(self.test_set["music_10"]) == 4874, f'len(test_set): {len(self.test_set["music_10"])}'
        assert len(self.test_set["music_15"]) == 4874, f'len(test_set): {len(self.test_set["music_15"])}'
        assert len(self.test_set["music_20"]) == 4874, f'len(test_set): {len(self.test_set["music_20"])}'
        assert len(self.test_set["nonspeech_0"]) == 4874, f'len(test_set): {len(self.test_set["nonspeech_0"])}'
        assert len(self.test_set["nonspeech_5"]) == 4874, f'len(test_set): {len(self.test_set["nonspeech_5"])}'
        assert len(self.test_set["nonspeech_10"]) == 4874, f'len(test_set): {len(self.test_set["nonspeech_10"])}'
        assert len(self.test_set["nonspeech_15"]) == 4874, f'len(test_set): {len(self.test_set["nonspeech_15"])}'
        assert len(self.test_set["nonspeech_20"]) == 4874, f'len(test_set): {len(self.test_set["nonspeech_20"])}'
        assert len(self.test_trials) == 37720, f'len(test_trials): {len(self.test_trials)}'

    def _parse_item(self, path, key):    
        for root, _, files in os.walk(path):
            for file in files:
                if '.wav' in file and len(file) == 9:                    
                    temp = os.path.join(root, file)
                    self.test_set[key].append(
                        TestItem(
                            path=temp,
                            key='/'.join(temp.split('/')[-3:])
                        )
                    )

    def _parse_trials(self, path):
        trials = []

        f = open(path) 
        for line in f.readlines():
            strI = line.split(' ')
            trials.append(
                TestTrial(
                    key1=strI[1].replace('\n', ''),
                    key2=strI[2].replace('\n', ''),
                    label=strI[0]
                )
            )
        
        return trials
