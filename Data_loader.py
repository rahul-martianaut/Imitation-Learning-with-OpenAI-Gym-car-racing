import torch
from PIL import Image
from torch.utils.data import Dataset
import numpy as np



class CarRacingDataset_RNN(Dataset):
    def __init__(self, images, labels, sequence_length, transform=None):
        self.images = images
        self.labels = labels
        self.sequence_length = sequence_length
        self.transform = transform

    def __getitem__(self, index):
        # Extract a sequence of frames
        start_idx = index
        end_idx = start_idx + self.sequence_length
        sequence = self.images[start_idx:end_idx]
        #sequence = np.transpose(sequence, (0, 3, 1, 2))
        label_sequence = self.labels[start_idx:end_idx]

        if len(sequence) < self.sequence_length:
            # If the sequence is shorter than the specified length, pad it
            padding_frames = [np.zeros_like(sequence[0])] * (self.sequence_length - len(sequence))
            sequence = np.concatenate([sequence, padding_frames])
            # You might need to handle padding for labels as well based on your requirement
            last_label = label_sequence[-1]
            padding_labels = [last_label] * (self.sequence_length - len( label_sequence))

            label_sequence = np.concatenate((label_sequence,padding_labels), axis=0)

        if self.transform:

            transformed_sequence = [self.transform(frame) for frame in sequence]
            # Leave the transformed frames as PyTorch tensors
            sequence = torch.stack(transformed_sequence)

        return sequence, label_sequence

    def __len__(self):
        return len(self.images) - self.sequence_length



class CarRacingDataset(Dataset):
    def __init__(self, images, labels, transform=None, grayscale = False):
        self.images = images
        self.labels = labels
        self.transform = transform
        self.grayscale = grayscale

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.fromarray(self.images[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        if self.grayscale:
            image = image.convert("L")

        if self.transform:
            image = self.transform(image)

        return image, label


