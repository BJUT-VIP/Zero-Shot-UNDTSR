import torch
from torch import nn
from torchvision.models.vgg import vgg16


class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()
        vgg = vgg16(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:5]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        print(vgg.features[:5])
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()
        self.L1_loss = nn.SmoothL1Loss()

    def forward(self, out_images, target_images):
        vgg_loss = self.L1_loss(self.loss_network(out_images), self.loss_network(target_images))
        image_loss = self.mse_loss(out_images, target_images)
        # print('image_loss:{:3f}  vgg_loss:{:3f} '.format(image_loss, vgg_loss))

        return image_loss + 0.5*vgg_loss

if __name__ == "__main__":
    g_loss = GeneratorLoss()
    print(g_loss)
