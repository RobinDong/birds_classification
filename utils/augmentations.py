import math
import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGE_SHAPE = (300, 300)


def generate_grid(h, w):
    x = torch.arange(0, h, dtype=torch.int)
    y = torch.arange(0, w, dtype=torch.int)
    grid = torch.stack([x.repeat(w), y.repeat(h, 1).t().contiguous().view(-1)], 1)
    return grid


class Augmentations(nn.Module):

    def __init__(self):
        super(Augmentations, self).__init__()
        self.gaussian_blur = nn.Conv2d(3, 3, (3, 3), padding=1, groups=3, bias=False).cuda()
        self.heavy_blur = nn.Conv2d(3, 3, (5, 5), padding=2, groups=3, bias=False).cuda()
        self.dropout = nn.Dropout(p=0.1).cuda()

    def _image_part(self, split, coordinates, index):
        coord = coordinates[index]
        part = IMAGE_SHAPE[0] // split
        return coord[0] * part, (coord[0] + 1) * part, coord[1] * part, (coord[1] + 1) * part

    def puzzle(self, image):
        raw_image = image.clone()
        split = 2
        coordinates = generate_grid(split, split)
        for index, offset in enumerate(torch.randperm(split*split)):
            nx1, nx2, ny1, ny2 = self._image_part(split, coordinates, index)
            ox1, ox2, oy1, oy2 = self._image_part(split, coordinates, offset)
            image[:, :, nx1:nx2, ny1:ny2] = raw_image[:, :, ox1:ox2, oy1:oy2]
            dim = torch.rand(1, device='cuda').round().long().data.item() + 2
            image[:, :, nx1:nx2, ny1:ny2] = torch.flip(raw_image[:, :, ox1:ox2, oy1:oy2], (dim,))
        return image

    def forward(self, image):
        # Gaussian blur
        '''self.gaussian_blur.weight.data.normal_(0.111, 0.02)
        image = self.gaussian_blur(image)

        gaussian_noise = torch.randn((3, IMAGE_SHAPE[0], IMAGE_SHAPE[1]), device='cuda')
        noise_scale = torch.rand(1, device='cuda') * 10
        image += gaussian_noise * noise_scale'''

        # Contrast 0.6 ~ 1.4
        gamma = 0.8 * torch.rand(1, device='cuda') + 0.6
        image *= gamma

        # Brightness -30 ~ 30
        brightness = 60 * torch.rand(1, device='cuda') - 30
        image += brightness

        image = torch.clamp(image, 0, 255)

        # Dropout or Heavy Gaussian Blur
        # rand = torch.rand(1, device='cuda')
        # if rand > 0.5:
        #    image = self.dropout(image)
        # else:
        #    self.heavy_blur.weight.data.normal_(0.04, 0.02)
        #    image = self.heavy_blur(image)'''

        # Flip horizontal or vertical for NCHW
        dim = torch.rand(1, device='cuda').round().long().data.item() + 2
        image = torch.flip(image, (dim,))

        # Rotate 90, 0 or -90
        k = torch.randint(-1, 2, (1,), device='cuda').item()
        image = torch.rot90(image, k=k, dims=[2, 3])

        # Rotate -90 ~ 90
        """angle = torch.rand(1, device='cuda') * 180 - 90
        elastic_scale = torch.rand((4), device='cuda')*0.4 + 0.8  # Also scale the image
        theta = torch.tensor([
                    [math.cos(angle) * elastic_scale[0], math.sin(-angle) * elastic_scale[1], 0],
                    [math.sin(angle) * elastic_scale[2], math.cos(angle) * elastic_scale[3], 0]
                ], device='cuda', dtype=torch.float)

        # 'theta' need to be Nx2x3 shape
        theta = theta.unsqueeze(0).repeat(image.size(0), 1, 1)
        grid = F.affine_grid(theta, image.size())
        image = F.grid_sample(image, grid)

        # Crop and resize
        orig_size = (image.size(2), image.size(3))
        x_offset = torch.rand(1, device='cuda') * image.size(2) * 0.2
        x_offset = x_offset.long()
        x_length = image.size(2) - x_offset
        x_length = x_length.long()
        y_offset = torch.rand(1, device='cuda') * image.size(3) * 0.2
        y_offset = y_offset.long()
        y_length = image.size(3) - x_offset
        y_length = y_length.long()
        image = image[:, :, x_offset:x_length, y_offset:y_length]
        image = F.interpolate(image, orig_size)"""

        # Randomly to gray
        '''rand = torch.rand(1, device='cuda')
        if rand > 0.5:
            gray = image.sum(dim=1) / image.shape[1]
            image = torch.stack([gray, gray, gray], dim=1)'''
        image = self.puzzle(image)
        return image


if __name__ == '__main__':
    import cv2
    files = ['/home/haodong/Downloads/samples/10010.House_Bunting_0_512_182_819_488_-_g_0061.jpg',
             '/home/haodong/Downloads/samples/2002.Mindanao_Bleeding-heart_0_26_366_815_1156_-_g_0090.jpg',
             '/home/haodong/Downloads/samples/8288.Dark-eyed_White-eye_0_221_280_607_666_-_e_0125.jpg']
    images = []
    for file in files:
        image = cv2.imread(file)
        image = cv2.resize(image, IMAGE_SHAPE)
        images.append(image)

    images = torch.Tensor(images).cuda().permute(0, 3, 1, 2).float()

    aug = Augmentations().cuda()
    imgs = aug(images).permute(0, 2, 3, 1).byte()

    for index, img in enumerate(imgs):
        cv2.imwrite('/home/haodong/Downloads/draw{}.jpg'.format(index), img.cpu().detach().numpy())
