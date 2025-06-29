import csv
import os
import torch
import random
import numpy as np
import soundfile as sf
import librosa

class AudioFolder(torch.utils.data.Dataset):
    def __init__(self, path):
        clips = []
        labels = []
        langs = set()

        with open(os.path.join(path, "data.csv"), "r") as f:
            reader = csv.reader(f)
            for row in reader:
                clips.append(os.path.join(path, row[0]))
                labels.append(row[1])
                langs.add(row[1])

        lang_id = dict()
        for idx, name in enumerate(langs):
            lang_id[name] = idx

        labels = list(map(lambda x: lang_id[x], labels))

        self.clips = clips
        self.labels = labels
        self.lang_id = lang_id

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        return self.clips[idx], self.labels[idx]
    
    def get_lang_id(self):
        return self.lang_id

class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, audio_folder,sample_rate, num_frames, min_num_frames, fixed):
        self.audio_folder = audio_folder
        self.sample_rate = sample_rate
        self.num_frames = num_frames
        self.min_num_frames = min_num_frames
        self.fixed = fixed
        if fixed:
            self.L_mem = [None] * len(self.audio_folder)

    def __len__(self):
        return len(self.audio_folder)

    def __getitem__(self, idx):
        path,label = self.audio_folder[idx]
        with sf.SoundFile(path, "r") as f:
            data = f.read()

            if f.samplerate != self.sample_rate:
                data = librosa.resample(data, orig_sr=f.samplerate, target_sr=self.sample_rate)

            def rand_L():
                return (
                    0
                    if data.shape[0] < self.min_num_frames
                    else random.randint(0, data.shape[0] - self.min_num_frames)
                )

            L = None

            if self.fixed:
                if self.L_mem[idx] is None:
                    self.L_mem[idx] = rand_L()
                L = self.L_mem[idx]
            else:
                L = rand_L()

            R = L + self.num_frames
            
            data = data[L:R]

            if len(data) < self.num_frames:
                padding = np.zeros(self.num_frames - len(data),dtype = data.dtype)
                data = np.concatenate((data, padding))

        return torch.from_numpy(data).float(), label

def load_dataset(*, path, sample_rate, test_size, num_frames, min_num_frames):
    audio_folder = AudioFolder(path)

    train_dataset,test_dataset = torch.utils.data.random_split(audio_folder,[1 - test_size, test_size])
    
    return {
        "train": AudioDataset(
            train_dataset,
            sample_rate=sample_rate,
            num_frames=num_frames,
            min_num_frames=min_num_frames,
            fixed=False,
        ),
        "test": AudioDataset(
            test_dataset,
            sample_rate=sample_rate,
            num_frames=num_frames,
            min_num_frames=min_num_frames,
            fixed=True
        ),
        "lang_id": audio_folder.get_lang_id(),
    }
