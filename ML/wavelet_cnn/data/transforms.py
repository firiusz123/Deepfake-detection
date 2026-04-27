from torchvision import transforms


def get_transforms(
    img_size=128,
    normalize=True,
    augment=False
):
    """
    Configurable transform pipeline for deepfake + wavelet model.

    Args:
        img_size (int): output resolution (64 / 128 / 256)
        normalize (bool): apply [-1, 1] normalization
        augment (bool): enable training augmentations
    """

    transform_list = []

    # =====================================================
    # RESIZE (critical for wavelet consistency)
    # =====================================================
    transform_list.append(
        transforms.Resize((img_size, img_size))
    )

    # =====================================================
    # DATA AUGMENTATION (ONLY FOR TRAINING)
    # =====================================================
    if augment:
        transform_list += [
            transforms.RandomHorizontalFlip(p=0.5),

            transforms.RandomApply([
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.05
                )
            ], p=0.5),

            transforms.RandomRotation(degrees=10),

            # important for deepfake robustness
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3)
            ], p=0.2),
        ]

    # =====================================================
    # TO TENSOR
    # =====================================================
    transform_list.append(transforms.ToTensor())

    # =====================================================
    # NORMALIZATION
    # =====================================================
    if normalize:
        transform_list.append(
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
        )

    return transforms.Compose(transform_list)
