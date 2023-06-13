from torchvision import datasets
from torchvision.transforms import ToTensor

train_data = datasets.MNIST(
    root = 'data',
    train = True,
    download = True,            
)
test_data = datasets.MNIST(
    root = 'data', 
    train = False,
)

for batch in train_data:
    images, labels = batch
    print(images)
    print(len(batch))
