import torch
from torch.utils.data import DataLoader, random_split, Subset
from torchvision.datasets import OxfordIIITPet
from torchvision import transforms 

def get_dataloaders(root="data_set",image_size = 224, batch_size = 32,
                    num_workers = 4, target_types = "category",train_split_size = 0.9,debug_subset_size = None):
    
    pin_memory = torch.cuda.is_available()

    
    transform = transforms.Compose([
        transforms.Resize((image_size,image_size)),
        transforms.ToTensor()
    ])

    train_val_dataset = OxfordIIITPet(
        root = root,
        split = "trainval",
        target_types = target_types,
        transform = transform,
        download = True
    )

    test_dataset = OxfordIIITPet(
        root = root,
        split = "test",
        target_types = target_types,
        transform = transform,
        download = True
    )

    train_size = int(train_split_size * len(train_val_dataset))
    val_size = len(train_val_dataset) - train_size

    train_dataset, val_dataset = random_split(
        train_val_dataset,
        [train_size,val_size]
    )

    # Trim datasets for debugging
    if debug_subset_size is not None:
        train_dataset = Subset(train_dataset, range(min(debug_subset_size, len(train_dataset))))
        val_dataset = Subset(val_dataset, range(min(debug_subset_size, len(val_dataset))))
        test_dataset = Subset(test_dataset, range(min(debug_subset_size, len(test_dataset))))

 

    train_loader = DataLoader(
        train_dataset,
        batch_size = batch_size,
        shuffle=True,
        num_workers = num_workers,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size = batch_size,
        shuffle=True,
        num_workers = num_workers,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size = batch_size,
        shuffle=True,
        num_workers = num_workers,
        pin_memory=pin_memory
    )

    return train_loader, val_loader, test_loader

