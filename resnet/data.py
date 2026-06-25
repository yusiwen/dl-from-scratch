from datasets import load_dataset
import torch
from torch.utils.data import Dataset


# The 40 CelebA attributes in their canonical order.
# This matches the attribute list order used by the eurecom-ds/celeba dataset on HF.
# We keep this at module level because it defines the mapping between the integer
# list returned by the HF dataset and the named attributes we reference by name.
CELEBA_ATTR_ORDER = [
    "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes",
    "Bald", "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair",
    "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin",
    "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones",
    "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard",
    "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline",
    "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair",
    "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace",
    "Wearing_Necktie", "Young",
]

# The 15 attributes we use for training.
# Why 15 instead of all 40? Fewer attributes makes the task simpler to debug
# and evaluate. We can always expand later. The chosen attributes cover a
# diverse set: facial features, hair, accessories, skin, and expression.
ATTRIBUTES = [
    "Smiling",
    "Male",
    "Young",
    "Eyeglasses",
    "Bald",
    "Heavy_Makeup",
    "Wearing_Lipstick",
    "Bangs",
    "Blond_Hair",
    "Black_Hair",
    "Chubby",
    "Mustache",
    "Wearing_Hat",
    "Pale_Skin",
    "Rosy_Cheeks",
]

# Pre-compute the column indices for our selected attributes.
# Why compute once at module level? Avoids repeated O(15*40) lookups
# every time __getitem__ is called.
_ATTR_INDICES = [CELEBA_ATTR_ORDER.index(a) for a in ATTRIBUTES]


class CelebADataset(Dataset):
    """
    CelebA dataset loaded via HuggingFace datasets library.

    The dataset is cached as Arrow files in ~/.cache/huggingface/datasets/
    (not individual image files on disk). Attributes are stored as a list of
    40 ints (0 or 1) in the canonical order defined above.

    We pre-extract labels for our selected 15 attributes lazily but only
    load the first `num_samples` images to control memory usage.

    This is a standard PyTorch Dataset subclass, meaning it works with
    torch.utils.data.DataLoader for batching and shuffling.
    """

    def __init__(self, attributes=None, num_samples=1000, transform=None):
        self.attributes = attributes or ATTRIBUTES
        self.transform = transform

        # Load only the first num_samples rows from the HF dataset.
        # The "train[:N]" split syntax reads N rows without materializing
        # the full 200K dataset. Internally, the HF datasets library
        # memory-maps the Arrow file and only decodes the rows we request.
        ds = load_dataset("eurecom-ds/celeba", split=f"train[:{num_samples}]")

        # Pre-extract labels into Python list + torch tensors.
        # We do this in __init__ rather than __getitem__ to avoid repeating
        # the attribute list indexing on every access. The images are stored
        # as PIL Images (not tensors) to save memory; transforms are applied
        # lazily in __getitem__.
        self.samples = []
        for item in ds:
            # Convert from HF's 0/1 to float 0.0/1.0 for BCEWithLogitsLoss.
            labels = [
                1.0 if item["attributes"][idx] == 1 else 0.0
                for idx in _ATTR_INDICES
            ]
            self.samples.append(((item["image"]), torch.tensor(labels, dtype=torch.float32)))

        # Release the HF dataset reference so memory can be reclaimed.
        del ds

    def __getitem__(self, idx):
        """
        Return (image, label) tuple where:
          - image is a transformed tensor (or PIL image if no transform)
          - label is a float32 tensor of shape (num_attributes,)
        """
        image_pil, labels = self.samples[idx]
        if self.transform:
            image = self.transform(image_pil)
        else:
            image = image_pil
        return image, labels

    def __len__(self):
        return len(self.samples)
