import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner, IidPartitioner


DATASET_NAMES = {
    "mnist": "mnist",
    "fmnist": "fashion_mnist",
    "cifar10": "cifar10",
    "cifar100": "cifar100",
}

TRANSFORMS = {
    "mnist": transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]),
    "fmnist": transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ]),
    "cifar10": transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ]),
    "cifar100": transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ]),
}


def _apply_transforms(batch, dataset_name: str):
    tfm = TRANSFORMS[dataset_name]
    img_key = "image" if "image" in batch else "img"
    batch["img"] = [tfm(img) for img in batch[img_key]]
    return batch


def _get_partitioner(num_clients: int, alpha, dataset_name: str):
    if alpha == "iid" or alpha is None:
        return IidPartitioner(num_partitions=num_clients)
    return DirichletPartitioner(
        num_partitions=num_clients,
        partition_by="label",
        alpha=float(alpha),
        min_partition_size=10,
        self_balancing=True,
        seed=42,
    )


_fds_cache = {}


def _get_fds(dataset_name: str, num_clients: int, alpha):
    cache_key = (dataset_name, num_clients, str(alpha))
    if cache_key not in _fds_cache:
        hf_name = DATASET_NAMES[dataset_name]
        partitioner = _get_partitioner(num_clients, alpha, dataset_name)
        _fds_cache[cache_key] = FederatedDataset(
            dataset=hf_name,
            partitioners={"train": partitioner},
        )
    return _fds_cache[cache_key]


def load_partition(
    partition_id: int,
    dataset_name: str,
    num_clients: int,
    alpha,
    batch_size: int = 32,
    test_split: float = 0.2,
):
    fds = _get_fds(dataset_name, num_clients, alpha)
    partition = fds.load_partition(partition_id)
    partition_splits = partition.train_test_split(test_size=test_split, seed=42)

    def apply_tfm(batch):
        return _apply_transforms(batch, dataset_name)

    train_ds = partition_splits["train"].with_transform(apply_tfm)
    test_ds = partition_splits["test"].with_transform(apply_tfm)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=False,
        collate_fn=_collate_fn,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        collate_fn=_collate_fn,
    )
    return train_loader, test_loader


def load_global_testset(dataset_name: str, num_clients: int, alpha, batch_size: int = 128):
    fds = _get_fds(dataset_name, num_clients, alpha)
    testset = fds.load_split("test")

    def apply_tfm(batch):
        return _apply_transforms(batch, dataset_name)

    testset = testset.with_transform(apply_tfm)
    return DataLoader(testset, batch_size=batch_size, shuffle=False, collate_fn=_collate_fn)


def _collate_fn(batch):
    imgs = torch.stack([item["img"] for item in batch])
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    return imgs, labels
