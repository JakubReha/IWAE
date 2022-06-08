import torchvision
import torch

class Binarizator(object):
    def __init__(self, inv=False):
        self.inv = inv
    def __call__(self, x):
        """Stochastic binarization in proportion to the pixel intensity"""
        x = torch.bernoulli(x)
        if self.inv:
            return (torch.abs(1-x)).to(x.dtype)
        else:
            return x.to(x.dtype)

def create_dataloaders(name, args):
    transform_mnist = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(), torchvision.transforms.Resize((28, 28)), Binarizator()])
    transform_omniglot = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(), torchvision.transforms.Resize((28, 28)), Binarizator(inv=True)])
    if name.lower() == "omniglot":
        dataset_train = torchvision.datasets.Omniglot(
            root="./data", download=True, background=True,
            transform=transform_omniglot
        )
        dataset_test = torchvision.datasets.Omniglot(
            root="./data", download=True, background=False,
            transform=transform_omniglot
        )
    elif name.lower() == "mnist":
        dataset_train = torchvision.datasets.MNIST(
            root="./data", download=True, train=True,
            transform=transform_mnist
        )
        dataset_test = torchvision.datasets.MNIST(
            root="./data", download=True, train=False,
            transform=transform_mnist
        )
    else:
        raise Exception('No such dataset')

    train_loader = torch.utils.data.DataLoader(dataset_train,
                                               batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset_test,
                                               batch_size=args.batch_size, shuffle=False)
    return train_loader, test_loader