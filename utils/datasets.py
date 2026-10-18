"""
cd fall-ballooms
python -m utils.datasets
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
import csv

from .preprocess import preprocess


class CirCoBioSAR(Dataset):
    """
    BioSAR CirCo version
    split = 'train', 'val', 'test'
    train = True -> randomly select chunk
    """
    def __init__(self, metadata_path='./datasets/BioSARdBScene-CirCo/metadata.csv', data_path = './datasets/BioSARdBScene-CirCo', split=None, train=False):
        self.data_path = Path(data_path)
        self.train = train
        self.items = []

        # load data from metadata.csv
        with open(metadata_path, newline="") as f:
            for row in csv.DictReader(f):

                # get only the heart (primary) from circo
                if row["organ"] != "Heart" or "Circo" not in row["primary_source"]:
                    continue 

                # split
                if split is not None and row["split"] != split:
                    continue

                # add items 
                self.items.append({
                    "audio_path": self.data_path / row["audio_path"],
                    "label": int(row["label"] == "abnormal"),
                    "organ": row["organ"],
                    "mixture_id": row["mixture_id"]
                })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]

        features = preprocess(str(item["audio_path"]), train=self.train)

        return (
            features,
            torch.tensor(item["label"], dtype=torch.float32),
            item["mixture_id"],
        )

class CirCo(CirCoBioSAR):
    """
    Clean CirCo version of CirCoBioSAR.
    Same rows and train/val/test split (from the BioSAR metadata.csv), but the audio
    is the original CirCo recording (primary_source) instead of the BioSAR mixture.
    """
    def __init__(self, metadata_path='./datasets/BioSARdBScene-CirCo/metadata.csv', data_path='./datasets/CirCo/training_data/training_data', split=None, train=False):
        super().__init__(metadata_path, data_path, split=split, train=train)

        # BioSAR mixture path -> clean CirCo file (same name as primary_source)
        primary = {}
        with open(metadata_path, newline="") as f:
            for row in csv.DictReader(f):
                primary[row["mixture_id"]] = Path(row["primary_source"]).name
        for item in self.items:
            item["audio_path"] = self.data_path / primary[item["mixture_id"]]

class BioSARHeart(Dataset):
    def __init__(self, 
                data_path='./datasets/BioSARDdBScene/', 
                metadata_path='datasets/BioSARDdBScene/metadata.csv', 
                split=None, 
                train=False):
        
        # make list of path to file
        self.data_path = Path(data_path)
        self.train = train
        self.items = [] # contain the paths

        # load data from metadata.csv
        with open(metadata_path, newline="") as f:
            for row in csv.DictReader(f):
                # get only the heart (primary)
                if row["organ"] != "Heart":
                    continue 

                # get split
                if split is not None and row["split"] != split:
                    continue

                # add items 
                self.items.append({
                    "audio_path": self.data_path / row["audio_path"],
                    "label": int(row["label"] == "abnormal"),
                    "organ": row["organ"],
                    "mixture_id": row["mixture_id"]
                })
        
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, index):
        # preprocess
        # feed to the DatasetLoader
        item = self.items[index]
    
        features = preprocess(str(item["audio_path"]), train=self.train)
    
        return (features,
                torch.tensor(item["label"], dtype=torch.float32),
                item["mixture_id"])


ORGANS = ["Bowel", "Heart", "Joint", "Lung"]
CLASS_NAMES = [f"{o}-{c}" for o in ORGANS for c in ("normal", "abnormal")]


class BioSARMulti(Dataset):
    """
    All organs of BioSAR, 8 classes: class = 2 * organ_index + (label == abnormal).
    Class names are in CLASS_NAMES.
    """
    def __init__(self,
                data_path='./datasets/BioSARDdBScene/',
                metadata_path='datasets/BioSARDdBScene/metadata.csv',
                split=None,
                train=False):
        self.data_path = Path(data_path)
        self.train = train
        self.items = []

        with open(metadata_path, newline="") as f:
            for row in csv.DictReader(f):
                if split is not None and row["split"] != split:
                    continue
                self.items.append({
                    "audio_path": self.data_path / row["audio_path"],
                    "label": 2 * ORGANS.index(row["organ"]) + int(row["label"] == "abnormal"),
                    "mixture_id": row["mixture_id"],
                })

    def __len__(self):
        return len(self.items)

    def labels(self):
        return [item["label"] for item in self.items]

    def __getitem__(self, index):
        item = self.items[index]
        features = preprocess(str(item["audio_path"]), train=self.train)
        return features, torch.tensor(item["label"], dtype=torch.long), item["mixture_id"]


if __name__ == "__main__":
    train_data = BioSARHeart(split='train')
    val_data = BioSARHeart(split='val')
    test_data = BioSARHeart(split='test')

    print(f"train: {len(train_data)}, val: {len(val_data)}, test: {len(val_data)}")
