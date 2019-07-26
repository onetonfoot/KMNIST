from albumentations import OneOf, IAAAdditiveGaussianNoise, GaussNoise, GridDistortion, IAAPiecewiseAffine, Blur, Rotate, Compose

noise = OneOf([
    IAAAdditiveGaussianNoise(),
    GaussNoise(),
], p=0.2)

distortion = OneOf([
    #     OpticalDistortion(),
    GridDistortion(distort_limit=0.4),
    IAAPiecewiseAffine(),
])

blur = OneOf([
    # MotionBlur(p=1),
    #     MedianBlur(blur_limit=7, p=1),
    Blur(blur_limit=3, p=1),
])

rotation = Rotate(limit=10)

aug = Compose([
    noise,
    distortion,
    blur,
    rotation
])
