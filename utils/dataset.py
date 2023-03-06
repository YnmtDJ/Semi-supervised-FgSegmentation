import os
import cv2
import torch.utils.data
import torchvision.transforms
from numpy import random


class GeneratorDataset(torch.utils.data.Dataset):
    def __init__(self, root, inputSize):
        super(GeneratorDataset, self).__init__()
        self.root = root
        self.tags = os.listdir(root)
#         if self.tags.count("Foliage"):
#             self.tags.remove("Foliage")
#             self.tags.remove("PeopleAndFoliage")
#             self.tags.remove("Snellen")
        self.input = []
        self.groundtruth = []
        self.background = []
        for tag in self.tags:
            input_dir = os.path.join(root, tag, "input")
            groundtruth_dir = os.path.join(root, tag, "groundtruth")
            background_dir = os.path.join(root, tag, 'background')
            for input_img in sorted(os.listdir(input_dir)):
                path = os.path.join(input_dir, input_img)
                self.input.append(path)
                background = random.choice(os.listdir(background_dir))
                path = os.path.join(background_dir, background)
                self.background.append(path)
            if os.path.exists(groundtruth_dir):
                for groundtruth_img in sorted(os.listdir(groundtruth_dir)):
                    path = os.path.join(groundtruth_dir, groundtruth_img)
                    self.groundtruth.append(path)

        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize(inputSize),
            # torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.brightness = torchvision.transforms.ColorJitter(brightness=0.8)

    def __len__(self):
        return len(self.input)

    def __getitem__(self, item):
        input = cv2.imread(self.input[item])
        input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
        input = self.transforms(input)
        # input = self.brightness(input)
        background = cv2.imread(self.background[item])
        background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
        background = self.transforms(background)
        groundtruth = []
        if len(self.groundtruth) > 0:
            groundtruth = cv2.imread(self.groundtruth[item])
            groundtruth = cv2.cvtColor(groundtruth, cv2.COLOR_BGR2GRAY)
            groundtruth = self.transforms(groundtruth)
            groundtruth[groundtruth <= 0.8] = 0
        return input, groundtruth, background