import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch , torchvision
from torchvision import datasets, transforms

def mnist_preprocessing_main():
    # 1. Define Data Transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # 2. Load MNIST Dataset
    train_data = torchvision.datasets.MNIST(root='data', train=True, download=True, transform=transform)
    test_data = torchvision.datasets.MNIST(root='data', train=False, download=True, transform=transform)

    # 3. Create Data Loaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)

    # 4. Define a function to show images
    def imshow(img):
        img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    def show_images():
        # Show Some Training Images
        dataiter = iter(train_loader)
        images, labels = next(dataiter)
        imshow(torchvision.utils.make_grid(images))
        print(' '.join(f'{train_data.classes[labels[j]]:5s}' for j in range(4)))

    def calculate_mean_std(loader):
        mean = 0.
        std = 0.
        for images, _ in loader:
            batch_samples = images.size(0)  # batch size (the last batch can have smaller size)
            images = images.view(batch_samples, images.size(1), -1)  # reshape to (batch_size, channels, height*width)
            mean += images.mean(2).mean(0)
            std += images.std(2).std(0)
        mean /= len(loader)
        std /= len(loader)
        return mean, std

    def show_mean():
        # Compute statistics
        mean, std = calculate_mean_std(train_loader)
        print(f'Mean: {mean}, Std: {std}')

    # Run functions
    show_images()
    show_mean()
