from albumentations import OneOf, IAAAdditiveGaussianNoise, GaussNoise, GridDistortion, IAAPiecewiseAffine, Blur, Rotate, Compose, RandomScale, Resize, PadIfNeeded, MedianBlur


noise = OneOf([
    IAAAdditiveGaussianNoise(),
    GaussNoise(),
], p=0.2)

distortion = OneOf([
    #     OpticalDistortion(),
    #     GridDistortion(distort_limit=0.4),
    IAAPiecewiseAffine(),
])

blur = OneOf([
    # MotionBlur(p=1),
    MedianBlur(blur_limit=7, p=1),
    Blur(blur_limit=3, p=1),
])

rotation = Rotate(limit=10)

scale = RandomScale(0.6)
pad = PadIfNeeded(64, 64, border_mode=1)
resize = Resize(64, 64)

aug = Compose([
    noise,
    #     distortion,
    blur,
    rotation,
    scale,
    pad,
    resize,
])
