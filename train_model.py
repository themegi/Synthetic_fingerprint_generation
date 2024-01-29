import torch
import torch.nn.parallel
import torch.utils.data
from typing import Tuple, List


#ustawienie etykiet
real_label = 1.
fake_label = 0.

#rozmiar wektora
z_size = 100


def train(netG: torch.nn.Module,
               netD: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               optimizerD: torch.optim.Adam,
               optimizerG:torch.optim.Adam,
               device: torch.device,
               criterion: torch.nn.Module,
               epochs: int) -> Tuple[List, List]:
    G_losses = []
    D_losses = []


    #wygenerowanie wektorów dla generatora
    fixed_noise = torch.randn(64, z_size, 1, 1, device=device)

    for epoch in range(epochs):
        for i, data in enumerate(dataloader, 0):
            #trening dyskryminatora na prawdziwych obrazach

            netD.zero_grad()
            #formatowanie partii
            real_cpu = data[0].to(device)
            batch_size = real_cpu.size(0)
            label = torch.full((batch_size,), real_label, dtype=torch.float, device=device)
            #ocena obrazów przez dyskryminator
            output = netD(real_cpu).view(-1)
            #obliczanie straty na prawdziwych obrazach
            errD_real = criterion(output, label)
            #obliczanie gradientu dla prawdziwych obrazów
            errD_real.backward()
            D_x = output.mean().item()


            #wygenerowanie wektorów dla generatora
            noise = torch.randn(batch_size, z_size, 1, 1, device=device)
            #wygenerowanie fałszywych obrazów
            fake = netG(noise)
            label.fill_(fake_label)
            #ocena obrazów przez dyskryminator
            output = netD(fake.detach()).view(-1)
            #obliczanie straty na fałszywych obrazach
            errD_fake = criterion(output, label)
            #obliczanie gradientu dla fałszywych i prawdziwych obrazów
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            #obliczanie straty dla fałszywych i prawdziwych obrazów
            errD = errD_real + errD_fake
            #ulepszenie sieci dyskryminatora
            optimizerD.step()


            #trening generatora

            netG.zero_grad()
            #ustawienie etykiet na prawdziwe
            label.fill_(real_label)
            #ponowna ocena obrazów przez dyskryminator
            output = netD(fake).view(-1)
            #obliczanie straty dla generatora
            errG = criterion(output, label)
            #obliczanie gradientu dla generatora
            errG.backward()
            D_G_z2 = output.mean().item()
            #ulepszenie sieci generatora
            optimizerG.step()


            G_losses.append(errG.item())
            D_losses.append(errD.item())


            #wyświetlanie postępu
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tStrata_dyskryminatora: %.4f\tStrata_generatora: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, epochs, i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

    return G_losses, D_losses
