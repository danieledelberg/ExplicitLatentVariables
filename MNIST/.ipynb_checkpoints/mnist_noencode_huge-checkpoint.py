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
        
        self.fc1 = nn.Linear(zdim, hdim)
        self.fc2 = nn.Linear(hdim, hdim)
        self.layer1 = nn.Linear(hdim, hdim)
        self.layer2 = nn.Linear(hdim, hdim)
        self.fc3 = nn.Linear(hdim, out_dim)
        
    #def encoder(self, x):
    #    h = F.relu(self.fc1(x))
    #    h = F.relu(self.fc2(h))
    #    return self.fc31(h), self.fc32(h) # mu, log_var
    
    #def sampling(self, mu, log_var):
    #    std = torch.exp(0.5*log_var)
    #    eps = torch.randn_like(std)
    #    return eps.mul(std).add_(mu) # return z sample
        
    def decoder(self, z):
        h = F.relu(self.fc1(z))
        h = F.relu(self.fc2(h))
        h = F.relu(self.layer1(h))
        h = F.relu(self.layer2(h))
        return torch.sigmoid(self.fc3(h)) # sigmoid, so we do not use logits
    
    def forward(self, z):
        #mu, log_var = self.encoder(x.view(-1, 784))
        #z = self.sampling(mu, log_var)
        return self.decoder(z)#, mu, log_var
    
class ZTracker(nn.Module):
    def __init__(self, zmu, zvar):
        super(ZTracker, self).__init__()
        self.zmu = zmu
        self.zvar = zvar
        # zvals shape: N x Zdim for each
        
        zmu_embed = nn.Embedding(zmu.shape[0], zmu.shape[1], sparse=True)
        zmu_embed.weight.data.copy_(zmu)
        zvar_embed = nn.Embedding(zvar.shape[0], zvar.shape[1], sparse=True)
        zvar_embed.weight.data.copy_(zvar)
        
        self.zmu_embed = zmu_embed
        self.zvar_embed = zvar_embed
        
    def get_zval(self, ind):
        zmu_val = self.zmu_embed(ind)
        zvar_val = self.zvar_embed(ind)
        return zmu_val, zvar_val
    
    def save(self, out_pkl):
        output_zmu = self.zmu_embed.weight.data.cpu().numpy()
        output_zvar = self.zvar_embed.weight.data.cpu().numpy()
        outputs = (output_zmu, output_zvar)
        pickle.dump(outputs, open(out_pkl, 'wb'))
        
        
def train_batch(model, y, zmu, zvar, optim, zoptim, beta):
    """Function that executes the forward and backpropagation functions for a single training batch"""
    optim.zero_grad()
    zoptim.zero_grad()
    model.train()

    z_mu, z_logvar, z, y_recon = run_batch(model, y, zmu, zvar)
    loss, gen_loss, kld = loss_function(z_mu, z_logvar, y, y_recon, beta)

    loss.backward()
    optim.step()
    zoptim.step()

    return loss.item(), gen_loss.item(), kld.item()


def run_batch(model, y, zmu, zvar):
    """Runs the forward function through the network, takes in z value and produces image"""
    B = y.size(0)
    z_mu, z_logvar = zmu, zvar
    std = torch.exp(0.5 * z_logvar)
    eps = torch.randn_like(std)
    z = z_mu + eps * std

    y_recon = model(z).view(B, -1)

    return z_mu, z_logvar, z, y_recon


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
    
    out_dir = 'noencoder_pretrained_notrain_HUGE'
    num_epochs = 100
    beta = 0
    batch_size = 128
    learning_rate = 0.01
    print('z training')
    zlearningrate = 0.00
    print(zlearningrate)
    zdim = 8
    
    # Load training sets
    train_dataset = TrainMNIST()
    test_dataset = TestMNIST()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    print('Using device: ', device)
    print('Using output directory: ', out_dir)
    
    pathExists = os.path.exists(out_dir)
    if not pathExists:
        os.makedirs(out_dir)
    else: 
        print('Output directory already exists, may be overwriting...')

    #print('Using pretrained initialization for latent space')
    #_mean_input, z_logvar_input = pickle.load(open('noencoder_zeros/z_vals.90.pkl', 'rb'))
    
    print('Using normal initialization for latent space')
    z_mean_input = np.random.normal(loc=0.0, scale=10, size=(len(train_dataset), zdim))
    z_logvar_input = np.ones((len(train_dataset), zdim)) * (-8)
    
    #print('Using zero initialization for latent space.')
    #z_mean_input = np.zeros((len(train_dataset), zdim))
    #z_logvar_input = np.ones((len(train_dataset), zdim)) * (-8)
    print('Mean of Z: ', np.mean(z_mean_input))
    print('Logvar of Z: ', np.mean(z_logvar_input))

            
    zmu_input = torch.tensor(z_mean_input, requires_grad=True)
    zvar_input = torch.tensor(z_logvar_input, requires_grad=True)

    zmu_input.to(device)
    zvar_input.to(device)

    ztracker = ZTracker(zmu_input, zvar_input).to(device)
    print('optimizer: sgd')
    z_optimizer = torch.optim.SGD(list(ztracker.parameters()), lr=zlearningrate)
    #z_optimizer = torch.optim.SparseAdam(list(ztracker.parameters()), lr=zlearningrate)


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

            zmu, zvar = ztracker.get_zval(ind)

            loss, gen_loss, kld = train_batch(model, y, zmu, zvar, optim, z_optimizer, beta)

            gen_loss_accum += gen_loss * B
            kld_accum += kld * B
            loss_accum += loss * B

        gen_loss_lst.append(gen_loss_accum / Nimg)
        kld_loss_lst.append(kld_accum / Nimg)
        loss_lst.append(loss_accum / Nimg)

        print('Epoch: {} Average gen loss = {:.6}, KLD = {:.6f}, total loss = {:.6f}'.format(epoch+1, gen_loss_accum/Nimg, kld_accum/Nimg, loss_accum/Nimg))
        if epoch % 10 == 0:
            out_z = '{}/z_vals.{}.pkl'.format(out_dir, epoch)
            ztracker.save(out_z)

    print('Training complete!')
    
    model.eval()
    print('Plotting...')
    plt.title('Loss Function')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(loss_lst)
    plt.show()

    output_zmu = ztracker.zmu_embed.weight.data.cpu().numpy()
    output_zlogvar = ztracker.zmu_embed.weight.data.cpu().numpy()
    
    colors = [tup[1] for tup in train_dataset]

    plt.title('First two Dimensions of Z')
    plt.scatter(output_zmu[:, 0], output_zmu[:, 1], c=colors, cmap='tab10', alpha=0.3, s=2)
    plt.colorbar()
    plt.show()

    from sklearn.decomposition import PCA
    pca = PCA(2)
    pc = pca.fit_transform(output_zmu)
    
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

        zmu, zvar = ztracker.get_zval(torch.tensor([i]).to(device))
        std = torch.exp(0.5 * zvar)
        eps = torch.randn_like(std)
        z = zmu + eps * std

        plt.title('Generated Image') 
        plt.imshow(model(z).reshape((28,28)).cpu().detach().numpy())
        plt.show()
    print('Done!')
    return 0

if __name__ == '__main__':
    main()