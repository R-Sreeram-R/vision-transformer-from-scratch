from torchvision.datasets import OxfordIIITPet #type:ignore

dataset = OxfordIIITPet(root="data", download=True)