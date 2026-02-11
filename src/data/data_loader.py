from torchvision.datasets import OxfordIIITPet # type: ignore

dataset = OxfordIIITPet(root="data_set", download=True)
