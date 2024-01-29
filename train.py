import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import data_setup, utils, train_model, model_build

#liczba epok
epochs = 5


#Optymalizatory
#współczynnik uczenia
lr = 0.0002
beta1 = 0.5

#pozostałe parametry
#liczba kanałów
colors = 3
#rozmiar wektora
z_size = 100
#rozmiar mapy cech generator
g_size = 64
#rozmiar mapy cech dyskryminatora
d_size = 64


#tryb: 1 - wykorzystanie GPU, 0 - wykorzystanie CPU
ngpu = 1


#ziarno dla powtarzalnych wyników
seed = 999
random.seed(seed)
torch.manual_seed(seed)
torch.use_deterministic_algorithms(True)


device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

#wywołanie dataloadera
dataloader = data_setup.dataloader

#inicjalizacja wag
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

#wywołanie funkcji generatora i dyskryminatora
netG = model_build.Generator(ngpu, z_size, g_size, colors).to(device)
netD = model_build.Discriminator(ngpu, d_size, colors).to(device)

#zaaplikowanie wag
netD.apply(weights_init)
netG.apply(weights_init)

#funckja straty
criterion = nn.BCELoss()

#inicjalizacja optymalizatorów
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

#trening generatora dyskryminatora
G_losses, D_losses = train_model.train(netG, netD, dataloader, optimizerD, optimizerG, device, criterion,
                                       epochs)

#utworzenie wykresu strat w trakcie uczenia
plt.figure(figsize=(10, 5))
plt.title("Strata generatora i dyskryminatora podczas treningu")
plt.plot(G_losses, label="G")
plt.plot(D_losses, label="D")
plt.xlabel("Iteracje")
plt.ylabel("Strata")
plt.legend()
plt.show()

#zapisanie wyników treningu w pliku
with open('params.txt', 'a') as f:
    f.write(f"Liczba epok:{epochs}, Strata_generatora {G_losses[-1]}, Strata_dyskrymiantora {D_losses[-1]} \n")

#zapisanie modelu
name = str(epochs) + "epochs.pth"
utils.save_model(netG, "models", name)



