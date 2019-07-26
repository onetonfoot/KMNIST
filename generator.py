from torchvision.utils import make_grid
from torchvision import transforms as T
import torch
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import random
import numpy as np


font_folder = Path("fonts/")
truetype = [ImageFont.truetype(font=str(font), size=64)
            for font in font_folder.glob("*/*.ttf")]
opentype = [ImageFont.truetype(font=str(font), size=64)
            for font in font_folder.glob("*/*.otf")]
fonts = [*truetype, *opentype]


def generate_images(char, fonts, width=64, height=64):

    images = []
    for font in fonts:
        image = Image.new("RGB", (width, height), (255, 255, 255))
        draw = ImageDraw.Draw(image)
        w, h = draw.textsize(char, font=font)

        w = (width - w) / 2
        h = (height - h * 1.5) / 2

        draw.text((w, h), char, fill='black', font=font)
        images.append(image)
    return images


class Generator:

    def __init__(self, chars, fonts, aug=None):
        self.chars = chars
        self.fonts = fonts
        self.aug = aug
        self.char2img = {}
        for char in self.chars:
            self.char2img[char] = generate_images(char, self.fonts)

    def sample(self, n, char=None):

        if char is None:
            # use random.sample if it don't want to repeat, however then cannot oversample so limits batch size
            chars = random.choices(list(self.char2img.keys()), k=n)
        else:
            chars = [char for _ in range(n)]

        imgs = []

        for char in chars:
            img = random.sample(self.char2img[char], 1)[0]
            if self.aug is not None:
                img = Image.fromarray(self.aug(image=np.array(img))["image"])
            imgs.append(img)

        return imgs, chars

    def make_grid(self, n, char=None):
        imgs, chars = self.sample(n, char)
        return T.ToPILImage()(make_grid(torch.stack([T.ToTensor()(img) for img in imgs])))
