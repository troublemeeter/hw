import torch.utils.data as Data
from dataset.prepare import prepare
import torch
import numpy as np
# 不知道数据精度不统一会不会有问题
# Note transforms.ToTensor() scales input images
# to 0-1 range


def LoadData(BATCH_SIZE):
    x,y = prepare()
    print('x has nan:',np.isnan(x).any())
    print('y has nan:',np.isnan(y).any())
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).long()
    print('tensor x and y of size: ',x.size(),y.size())
    dataset = Data.TensorDataset(x, y)

    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # 把 dataset 放入 DataLoader
    train_loader = Data.DataLoader(
        dataset=train_dataset,      # torch TensorDataset format
        batch_size=BATCH_SIZE,      # mini batch size
        shuffle=True,               # 要不要打乱数据 (打乱比较好)
        num_workers=3,              # 多线程来读数据
    )

    test_loader = Data.DataLoader(
        dataset=test_dataset,      # torch TensorDataset format
        batch_size=BATCH_SIZE,      # mini batch size
        shuffle=True,               # 要不要打乱数据 (打乱比较好)
        num_workers=3,              # 多线程来读数据
    )

    # Checking the dataset
    for feature, labels in train_loader:  
        print('Checking DataLoader ...')
        print('feature dimensions:', feature.shape)
        print('labels exmaple:', labels[:10])
        break

    return train_loader,test_loader