from torch import nn


class Generator(nn.Module):
    def __init__(self, ngpu, z_size: int, g_size: int, colors: int) -> None:
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.ConvTranspose2d(z_size, g_size * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(g_size * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(g_size * 8, g_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(g_size * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(g_size * 4, g_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(g_size * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(g_size * 2, g_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(g_size),
            nn.ReLU(True),

            nn.ConvTranspose2d(g_size, colors, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, ngpu, d_size: int, colors: int) -> None:
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(

            nn.Conv2d(colors, d_size, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(d_size, d_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d_size * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(d_size * 2, d_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d_size * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(d_size * 4, d_size * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d_size * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(d_size * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
