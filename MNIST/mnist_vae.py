import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os

# Custom datasets for train and test sets, used to extract the index with the actual content of the images.
class TrainMNIST(Dataset):
    def __init__(self):
        self.mnist = datasets.MNIST(root='./mnist_data',
                                        download=True,
                                        train=True,
                                        transform=transforms.ToTensor())
        
    def __getitem__(self, index):
        data, label = self.mnist[index]  
        return data.flatten(), label, index

    def __len__(self):
        return len(self.mnist)

class TestMNIST(Dataset):
    def __init__(self):
        self.mnist = datasets.MNIST(root='./mnist_data',
                                        download=True,
                                        train=False,
                                        transform=transforms.ToTensor())
        
    def __getitem__(self, index):
        data, _ = self.mnist[index]  
        return data.flatten(), index

    def __len__(self):
        return len(self.mnist)



class VAE(nn.Module):
    def __init__(self, zdim, out_dim, hdim):
        super(VAE, self).__init__()
        
        self.hdim = hdim 
        
        self.fc1 = nn.Linear(out_dim, hdim)
        self.fc2 = nn.Linear(hdim, hdim)
        self.fc3a = nn.Linear(hdim, zdim)
        self.fc3b = nn.Linear(hdim, zdim)
        
        self.fc4 = nn.Linear(zdim, hdim)
        self.fc5 = nn.Linear(hdim, hdim)
        self.fc6 = nn.Linear(hdim, out_dim)
        
    def encoder(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc3a(h), self.fc3b(h) # mu, log_var
    
    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu) # return z sample
        
    def decoder(self, z):
        h = F.relu(self.fc4(z))
        h = F.relu(self.fc5(h))
        return torch.sigmoid(self.fc6(h)) # sigmoid, so we do not use logits
    
    def forward(self, x):
        mu, log_var = self.encoder(x.view(-1, 784))
        z = self.sampling(mu, log_var)
        return self.decoder(z), mu, log_var
   

def train_batch(model, y, optim, beta):
    """Function that executes the forward and backpropagation functions for a single training batch"""
    optim.zero_grad()
    model.train()

    z_mu, z_logvar, y_recon = run_batch(model, y)
    loss, gen_loss, kld = loss_function(z_mu, z_logvar, y, y_recon, beta)

    loss.backward()
    optim.step()

    return loss.item(), gen_loss.item(), kld.item()


def run_batch(model, y):
    """Runs the forward function through the network, takes in z value and produces image"""
    B = y.size(0)
    y_recon, z_mu, z_logvar = model(y)

    return z_mu, z_logvar, y_recon


def loss_function(z_mu, z_logvar, y, y_recon, beta):
    """Calculates loss functions"""
    B = y.size(0)
    # use without logits because we do force outputs with sigmoids
    gen_loss = F.binary_cross_entropy(y_recon, y.view(B, -1), reduction='sum')
    # KLD vs. N(0,1) distribution
    kld = -0.5 * torch.sum(1 + z_logvar - z_mu.pow(2) - z_logvar.exp())
    loss = gen_loss + beta * kld

    return loss, gen_loss, kld


def main():
    
    out_dir = 'large_encoder'
    num_epochs = 100
    beta = 1
    batch_size = 128
    learning_rate = 0.01
    zdim = 10
    
    # Load training sets
    train_dataset = TrainMNIST()
    test_dataset = TrainMNIST() #Use test as train
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=60000, shuffle=False)

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    print('Using device: ', device)
    print('Using output directory: ', out_dir)
    
    pathExists = os.path.exists(out_dir)
    if not pathExists:
        os.makedirs(out_dir)
    else: 
        print('Output directory already exists, may be overwriting...')


    model = VAE(zdim, 28*28, 256)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)

    Nimg = len(train_dataset)
    print('Training...')
    gen_loss_lst = []
    kld_loss_lst = []
    loss_lst = []
    for epoch in range(num_epochs):

        gen_loss_accum = 0
        kld_accum = 0
        loss_accum = 0

        for minibatch in train_loader:
            ind = minibatch[-1].to(device)
            y = minibatch[0].to(device)
            B = len(ind)

            loss, gen_loss, kld = train_batch(model, y, optim, beta)

            gen_loss_accum += gen_loss * B
            kld_accum += kld * B
            loss_accum += loss * B

        gen_loss_lst.append(gen_loss_accum / Nimg)
        kld_loss_lst.append(kld_accum / Nimg)
        loss_lst.append(loss_accum / Nimg)

        print('Epoch: {} Average gen loss = {:.6}, KLD = {:.6f}, total loss = {:.6f}'.format(epoch+1, gen_loss_accum/Nimg, kld_accum/Nimg, loss_accum/Nimg))

    print('Training complete!')
    print('Plotting...')
    plt.title('Loss Function')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(loss_lst)
    plt.show()

    print('Generating Z Values...')
    colors = [tup[1] for tup in train_dataset]
    # Very hacky way to read all values in at once
    for i in test_loader:
        zmu, zlogvar = model.encoder(i[0].to(device))
        
    zmu = zmu.cpu().detach().numpy()
    
    plt.scatter(zmu[:, 0], zmu[:, 1], c=colors, cmap='tab10', alpha=0.3, s=2)
    plt.colorbar()
    plt.show()
    
    from sklearn.decomposition import PCA
    pca = PCA(2)
    pc = pca.fit_transform(zmu)
    
    plt.title('PCA of Z space')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.scatter(pc[:, 0], pc[:, 1], c=colors, cmap='tab10', alpha=0.3, s=2)
    plt.colorbar()
    plt.show()
    
    for i in range(20):
        plt.title('Original Image')
        plt.imshow(train_dataset[i][0].reshape((28,28)))
        plt.show()

        plt.title('Generated Image')
        plt.imshow(model((train_dataset[i][0]).to(device))[0].reshape((28,28)).cpu().detach().numpy())
        plt.show()
        
    print('Done!')
    return 0

if __name__ == '__main__':
    main()