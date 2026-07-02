# Reuse ResNet34's data pipeline — same CelebA dataset, same transforms.
from cv.resnet34.data import CelebADataset, CELEBA_ATTR_ORDER, train_transform, eval_transform  # noqa: F401
