import pickle
import torch.utils.data as Data
from dataset import MoseiDataset
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_dataloader(config):
    if config["dataset"] == "mosi":
        with open('./data/mosi_unaligned.pkl', 'rb') as f:
            data_dic = pickle.load(f)
    elif config["dataset"] == "mosei":
        with open('./data/mosei_unaligned.pkl', 'rb') as f:
            data_dic = pickle.load(f)
    else:
        raise ValueError("Invalid dataset !")

    train_dataset = MoseiDataset(data_dic, device, 'train')
    valid_dataset = MoseiDataset(data_dic, device, 'valid')
    test_dataset = MoseiDataset(data_dic, device, 'test')

    train_loader = Data.DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    valid_loader = Data.DataLoader(valid_dataset, batch_size=1, shuffle=True)
    test_loader = Data.DataLoader(test_dataset, batch_size=1)

    print("\tNumber of train samples: %d\n\tNumber of valid samples: %d\n\tNumber of test samples: %d"
          % (len(train_dataset), len(valid_dataset), len(test_dataset)))

    return train_loader, valid_loader, test_loader
