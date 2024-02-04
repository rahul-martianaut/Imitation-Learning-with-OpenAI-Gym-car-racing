import numpy as np
import matplotlib.pyplot as plt
import cv2
import torchvision.transforms as transforms
import torch


def visualize(data_loader, num_samples=30):

    for images, actions in data_loader:

        rows = int(np.sqrt(num_samples))
        cols = int(np.ceil(num_samples / rows))
        plt.figure(figsize=(10, 10))

        for i in range(num_samples):
            plt.subplot(rows, cols, i + 1)
            observation = images[i]
            action = actions[i].tolist()
            plt.imshow(np.transpose(observation, (1, 2, 0)), interpolation='nearest')
            plt.title(f"Action: {action}", fontsize=7)
            plt.axis('off')

        plt.show()
        break


class ChangeColorTransform:

    def __call__(self, img):

        img = np.array(img)
        img = np.transpose(img, (1, 2, 0))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        mask_grey = cv2.inRange(hsv, (0, 0, 0.2), (180, 0.1, 0.5))  # Mask for grey color

        img[mask_grey > 0] = (50, 0.5, 0.3)  # Grey to Brown color

        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img).float()

        return img

class RandomCropAndRotation:
    def __init__(self, crop_size=(80, 80), rotation_angle=20):
        self.crop_size = crop_size
        self.rotation_angle = rotation_angle

    def __call__(self, img):
        # Random crop
        i, j, h, w = transforms.RandomCrop.get_params(img, output_size=self.crop_size)
        img = transforms.functional.crop(img, i, j, h, w)

        # Random rotation
        angle = np.random.uniform(-self.rotation_angle, self.rotation_angle)
        img = transforms.functional.rotate(img, angle)


        return img


