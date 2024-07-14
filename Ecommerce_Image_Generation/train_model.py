import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
from torchvision import datasets, transforms

class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

latent_dim = 100
image_dim = 28 * 28 * 1  # for MNIST dataset
batch_size = 64
learning_rate = 0.0002
epochs = 5 # preferablly higher like 50 to generate better Images

generator = Generator(latent_dim, image_dim)
discriminator = Discriminator(image_dim)

optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate)
optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate)

adversarial_loss = nn.BCELoss()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

dataset = datasets.MNIST(root='data', train=True, transform=transform, download=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Training
for epoch in range(epochs):
    for batch in dataloader:
        real_images, _ = batch
        real_images = real_images.view(real_images.size(0), -1)

        optimizer_D.zero_grad()
        real_labels = torch.ones(real_images.size(0), 1)
        fake_labels = torch.zeros(real_images.size(0), 1)

        outputs = discriminator(real_images)
        d_loss_real = adversarial_loss(outputs, real_labels)

        z = torch.randn(real_images.size(0), latent_dim)
        fake_images = generator(z)
        outputs = discriminator(fake_images)
        d_loss_fake = adversarial_loss(outputs, fake_labels)

        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_D.step()

        optimizer_G.zero_grad()
        z = torch.randn(real_images.size(0), latent_dim)
        fake_images = generator(z)
        outputs = discriminator(fake_images)

        g_loss = adversarial_loss(outputs, real_labels)
        g_loss.backward()
        optimizer_G.step()

    print(f"Epoch [{epoch}/{epochs}]  Loss D: {d_loss.item()}, loss G: {g_loss.item()}")

torch.save(generator.state_dict(), 'generator.pth')
