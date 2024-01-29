import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms

#wymiary obrazów wejściowych
image_size = 64
#wielkość partii treningowej
batch_size = 128

#należy skonfigurować ścieżkę do treningowej bazy danych
dataroot = "./dataset/"
dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True)

