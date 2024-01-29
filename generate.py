import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
import utils


#ilość odcisków do wygenerowania
fng_quant = 48


nz = 100
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
fixed_noise = torch.randn(fng_quant, nz, 1, 1, device=device)

#należy wybrać model, który ma zostać załadowany
loaded = utils.load_model("socofing.pth")

img_list = []

fake = loaded(fixed_noise).detach().cpu()

#wygenerowane odciski zostaną zapisane w formacie .bmp w wybranym katalogu
for i in range(fng_quant):
    img = vutils.save_image(fake[i],f"./.../{i}.bmp", "bmp", normalize=True, padding=0)


#wyświetlenie odcisków w formie siatki
img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
plt.imshow(np.transpose(img_list[-1],(1,2,0)))

plt.show()
