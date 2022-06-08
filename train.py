from dataloaders import create_dataloaders
import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torchvision
from model import IWAE

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--k', type=int, default=5)
parser.add_argument('--batch-size', type=int, default=20,
                    help='batch size for training')
parser.add_argument('--epochs', type=int, default=100,
                    help='number of epochs to train')
parser.add_argument('--name', type=str, default="debug",
                    help='model name')
parser.add_argument('--no-cuda', action='store_true',
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--type', type=str, default="iwae",
                    help='iwae/vae')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
torch.manual_seed(args.seed)

if __name__ == '__main__':
    logdir = "./runs/" + args.name
    writer = SummaryWriter(logdir)
    dl_train, dl_test = create_dataloaders("mnist", args)
    model = IWAE(args.k).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for e in tqdm(range(args.epochs)):
        train_loss = 0.0
        model.train()
        for data, _ in dl_train:
            data = data.to(device)
            optimizer.zero_grad()
            dparams = model(data)
            target = dparams[-1]
            loss = model.compute_loss(data, dparams)[args.type]
            loss.backward()
            optimizer.step()
            train_loss += loss.item()/len(dl_train)
        writer.add_scalar('Loss/train', train_loss, e)
        valid_loss = 0.0
        model.eval()
        with torch.no_grad():
            for data, _ in dl_test:
                data = data.to(device)
                dparams = model(data)
                target = dparams[-1]
                loss = model.compute_loss(data, dparams)[args.type]
                valid_loss = loss.item()/len(dl_test)
        val_batch_im = torchvision.utils.make_grid(target.reshape(-1, 1, 28, 28))
        writer.add_image('im_generated', val_batch_im, e)
        val_batch_im = torchvision.utils.make_grid(data)
        writer.add_image('im_orig', val_batch_im, e)
        writer.add_scalar('Loss/valid', valid_loss, e)
    writer.close()
