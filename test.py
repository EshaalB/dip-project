from src.dataset import ImageDataset
import numpy as np

ds = ImageDataset("data")
gray, color = ds[0]

print("Gray range:", gray.min().item(), gray.max().item())
print("Color range:", color.min().item(), color.max().item())
print("Gray shape:", gray.shape)
print("Color shape:", color.shape)
