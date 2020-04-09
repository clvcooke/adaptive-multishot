import numpy as np
import os
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_x, data_y):
        self.data_x = data_x
        self.data_y = data_y
        if data_y.dtype is torch.float16:
            self.convert = True
        else:
            self.convert = False

    def __getitem__(self, index):
        image = self.data_x[index]
        label = self.data_y[index]

        if self.convert:
            return image.float(), label.float()
        else:
            return image.float(), label


def load_progress(path, desc=''):
    try:
        mmap_array = np.load(path, mmap_mode='r')
        array = np.empty_like(mmap_array, dtype=np.float16)
        block_size = 2
        n_blocks = int(np.ceil(mmap_array.shape[0] / block_size))
        for b in tqdm(range(n_blocks), desc=desc):
            array[b * block_size:(b + 1) * block_size] = mmap_array[b * block_size:(b + 1) * block_size]
    finally:
        del mmap_array
    return array


def shift_data(shift_code, data):
    if shift_code == '':
        return data
    # if we are going to shift  we can use N S E W along with a number (1-10)
    direction = shift_code[0].upper()
    amnt = int(shift_code[1:])
    assert direction in ['N', 'S', 'E', 'W']
    # we need to do a reshape first
    data = np.reshape(data, list(data.shape[0:1]) + [3, 15, 15, 256, 256])
    # now we need to cutoff the data on the correct axis
    if direction == 'S':
        data = data[:, :, amnt:]
    elif direction == 'N':
        data = data[:, :, :-amnt]
    elif direction == 'E':
        data = data[:, :, :, :-amnt]
    else:
        data = data[:, :, :, amnt:]
    # reshape the data back to the proper format
    data = np.reshape(data, [-1, 675 - amnt*15*3, 256, 256])
    return data


def get_train_val_loader(config, pin_memory, num_workers=1):
    data_dir = '/hddraid5/data/colin/'
    batch_size = config.batch_size
    if str(config.task).lower() == 'hela':
        mmap = False
        train_x_path = os.path.join(data_dir, 'ctc', 'train_x_norm.npy')
        train_y_path = os.path.join(data_dir, 'ctc', f'new_nuc_train_kb7.npy')
        val_x_path = os.path.join(data_dir, 'ctc', 'val_x_norm.npy')
        val_y_path = os.path.join(data_dir, 'ctc', f'new_nuc_val_kb7.npy')
    else:
        mmap = False
        train_x_path = os.path.join(data_dir, 'new_pan_data', 'train_x_final.npy')
        train_y_path = os.path.join(data_dir, 'new_pan_data', f'train_y_final.npy')
        val_x_path = os.path.join(data_dir, 'new_pan_data', 'val_x_final.npy')
        val_y_path = os.path.join(data_dir, 'new_pan_data', f'val_y_final.npy')

    # pytorch says channels fist
    if mmap:
        train_x_npy = shift_data(config.shift, np.load(train_x_path, mmap_mode='r'))
        train_x = torch.from_numpy(train_x_npy)
        train_y = torch.from_numpy(np.load(train_y_path, mmap_mode='r'))
        val_x_npy = shift_data(config.shift, np.load(val_x_path, mmap_mode='r'))
        val_x = torch.from_numpy(val_x_npy)
        val_y = torch.from_numpy(np.load(val_y_path, mmap_mode='r'))
    else:
        train_x_npy = shift_data(config.shift, load_progress(train_x_path, 'loading train x'))
        train_x = torch.from_numpy(train_x_npy)
        train_y = torch.from_numpy(load_progress(train_y_path, 'loading train_y'))
        val_x_npy = shift_data(config.shift, load_progress(val_x_path, 'loading val_x'))
        val_x = torch.from_numpy(val_x_npy)
        val_y = torch.from_numpy(load_progress(val_y_path, 'loading val_y'))
    num_leds = train_x_npy.shape[1]
    train_dataset = CustomDataset(train_x, train_y)
    val_dataset = CustomDataset(val_x, val_y)

    train_idx, valid_idx = list(range(train_x.shape[0])), list(range(val_x.shape[0]))

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    valid_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, sampler=valid_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return train_loader, valid_loader, num_leds
