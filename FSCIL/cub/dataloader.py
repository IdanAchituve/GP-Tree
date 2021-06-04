from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
import logging
from utils import *


def generate_val_indices_set(root='./dataset/CUB_200_2011', val_pct=.067,
                             base_classes=None, write_path='./val_indices'):

    if base_classes is None:
        base_classes = np.arange(100)

    set_seed(42)

    train_test_df = pd.read_csv(os.path.join(root, 'train_test_split.txt'), sep=' ',
                                header=None, names=['image_id', 'set'])
    label_df = pd.read_csv(os.path.join(root, 'image_class_labels.txt'), sep=' ',
                           header=None, names=['image_id', 'label'])
    label_df['label'] = label_df['label'] - 1

    # read file with image path
    classes_df = pd.read_csv(os.path.join(root, 'images.txt'), sep=' ', header=None, names=['image_id', 'image_path'])

    # merge data
    df = pd.concat([classes_df, train_test_df, label_df], axis=1, join="inner")

    # remove duplicate columns
    _, i = np.unique(df.columns, return_index=True)
    df = df.iloc[:, i]

    # take only base classes from train set
    train_base_df = df[df.set == 1]
    train_base_df = train_base_df[train_base_df.label.isin(base_classes)]

    # take last 2 cloumns
    X_train, X_val = train_test_split(train_base_df[['image_id', 'image_path']],
                                       random_state=42, test_size=val_pct,
                                       stratify=train_base_df[['label']])

    X_train = X_train.sort_values(by=['image_id'])
    X_val = X_val.sort_values(by=['image_id'])

    # write to file
    X_val.to_csv(write_path, header=False, sep=' ', index=False)


class CUB200(Dataset):
    def __init__(self, logger, root='./dataset/CUB_200_2011', dataset='train',
                 transform=None, c_way=5, k_shot=5, fix_class=None, base_class=0,
                 fine_label=True, val_indices_path='./val_indices'):

        self.name = 'NC_CUB200'
        self._dataset = dataset
        self._logger = logger
        self._root = os.path.expanduser(root)
        self._transform = transform
        self._exts = ['.jpg', '.jpeg', '.png']
        self.val_indices_path = val_indices_path

        self._pre_operate(self._root)

        self._class = 100

        self._c_way = c_way
        self._k_shot = k_shot
        self._fix_class = fix_class
        self._base_class = base_class
        self._few_shot()

    def text_read(self,file):
        with open(file,'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                lines[i] = line.strip('\n')
        return lines

    def list2dict(self,list):
        dict = {}
        for l in list:
            s = l.split(' ')
            id = int(s[0])
            cls = s[1]
            if id not in dict.keys():
                dict[id] = cls
            else:
                raise EOFError('The same ID can only appear once')
        return dict

    def _pre_operate(self, root):
        image_file = os.path.join(root,'images.txt')
        split_file = os.path.join(root,'train_test_split.txt')
        class_file = os.path.join(root,'image_class_labels.txt')
        id2image = self.list2dict(self.text_read(image_file))
        id2train = self.list2dict(self.text_read(split_file))  # 1: train images; 0: test iamges
        id2class = self.list2dict(self.text_read(class_file))
        train_idx = []
        test_idx  = []

        id2image_val = self.list2dict(self.text_read(self.val_indices_path))
        val_idx  = []

        for k in sorted(id2train.keys()):
            if id2train[k] == '1':
                if k in id2image_val.keys():
                    val_idx.append(k)
                else:
                    train_idx.append(k)
            else:
                test_idx.append(k)

        self.images = []
        self.labels = []
        if self._dataset == 'train':
            for k in train_idx:
                image_path = os.path.join(root,'images',id2image[k])
                self.images.append(image_path)
                self.labels.append(int(id2class[k])-1)

        elif self._dataset == 'val':
            for k in val_idx:
                image_path = os.path.join(root,'images',id2image[k])
                self.images.append(image_path)
                self.labels.append(int(id2class[k])-1)

        elif self._dataset == 'test':
            for k in test_idx:
                image_path = os.path.join(root,'images',id2image[k])
                self.images.append(image_path)
                self.labels.append(int(id2class[k])-1)

    def _few_shot(self):
        self._data = []

        if not self._fix_class:
            np.random.seed(0)  # random select classes
            classes = np.random.choice(self._class, size=self._c_way, replace=False)
            self._fix_class = list(classes)
        if self._logger:
            self._logger.info('select CUB200 classes : {} , train = {}'.
                  format(self._fix_class, self._dataset))

        if self._dataset == 'train':
            select_index = list()
            new_label = list()

            for i,l in enumerate(self._fix_class):
                ind = list(np.where(l==np.array(self.labels))[0])
                np.random.seed(1)  # random select pictures
                try:
                    random_ind = np.random.choice(ind, self._k_shot, replace=False)
                except:
                    random_ind = np.random.choice(ind, len(ind), replace=False)
                select_index.extend(random_ind)
                new_label.extend([i+self._base_class]*len(random_ind))
            for i in select_index:
                self._data.append(self.images[i])
            self._label = new_label
            #self._logger.info(self._data)
            #self._logger.info(self._label)

        else:
            select_index = list()
            new_label = list()
            for i,l in enumerate(self._fix_class):
                ind = list(np.where(l==np.array(self.labels))[0])
                select_index.extend(ind)
                new_label.extend([i+self._base_class]*len(ind))
            for i in select_index:
                self._data.append(self.images[i])
            self._label = new_label
        if self._logger:
            self._logger.info('the number of samples in %s: %d'%(self._dataset, len(new_label)))

    def __getitem__(self, idx):
        img = Image.open(self._data[idx]).convert('RGB')
        label = self._label[idx]
        if self._transform:
            img = self._transform(img)
        return img, label, idx

    def __len__(self):
        return len(self._label)


class CUB200_Indexed(Dataset):
    def __init__(self, logger, root='./dataset/CUB_200_2011', dataset='train',
                 transform=None, session=1, val_indices_path='./val_indices', index_path='./index_list'):

        self._dataset = dataset
        self._logger = logger
        self._data = []
        self._label = []
        self._root = os.path.expanduser(root)
        self._transform = transform
        self.val_indices_path = val_indices_path

        self.session = session
        self.index_path = index_path
        self.collect()

    def text_read(self,file):
        with open(file,'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                lines[i] = line.strip('\n')
        return lines

    def list2dict(self,list):
        dict = {}
        for l in list:
            s = l.split(' ')
            id = int(s[0])
            cls = s[1]
            if id not in dict.keys():
                dict[id] = cls
            else:
                raise EOFError('The same ID can only appear once')
        return dict

    def collect(self):
        self.id2image_val = list(self.list2dict(self.text_read(self.val_indices_path)).values())
        lbl_to_count = {}

        if self._dataset == 'train':
            img_path = self.text_read(self.index_path + '/session_' + str(self.session) + '.txt')
            for img in img_path:
                img_wo_root = "/".join(img.split('/')[2:])
                if img_wo_root not in self.id2image_val:
                    img_label = int(img.split('/')[2].split('.')[0]) - 1
                    self._label.append(img_label)
                    self._data.append(self._root + "/" + "/".join(img.split('/')[1:]))
                    lbl_to_count[img_label] = 1 if img_label not in list(lbl_to_count.keys()) \
                        else lbl_to_count[img_label] + 1
            if self._logger:
                self._logger.info("Classes session %s, frequencies %s: " % (str(self.session), self._dataset))
                self._logger.info(lbl_to_count)

        elif self._dataset == 'val':
            img_path = self.text_read(self.index_path + '/session_' + str(self.session) + '.txt')
            for img in img_path:
                img_wo_root = "/".join(img.split('/')[2:])
                if img_wo_root in self.id2image_val:
                    img_label = int(img.split('/')[2].split('.')[0]) - 1
                    self._label.append(img_label)
                    self._data.append(self._root + "/" + "/".join(img.split('/')[1:]))
                    lbl_to_count[img_label] = 1 if img_label not in list(lbl_to_count.keys()) \
                        else lbl_to_count[img_label] + 1
            if self._logger:
                self._logger.info("Classes session %s, frequencies %s: " % (str(self.session), self._dataset))
                self._logger.info(lbl_to_count)

        elif self._dataset == 'test':
            for i in range(1, self.session + 1):
                img_path = self.text_read(self.index_path + '/test_' + str(i) + '.txt')
                for img in img_path:
                    img_label = int(img.split('/')[2].split('.')[0]) - 1
                    self._label.append(img_label)
                    self._data.append(self._root + "/" + "/".join(img.split('/')[1:]))
                    lbl_to_count[img_label] = 1 if img_label not in list(lbl_to_count.keys()) \
                        else lbl_to_count[img_label] + 1
            if self._logger:
                self._logger.info("Classes session %s, frequencies %s: " % (str(self.session), self._dataset))
                self._logger.info(lbl_to_count)

    def __getitem__(self, idx):
        img = Image.open(self._data[idx]).convert('RGB')
        label = self._label[idx]
        if self._transform:
            img = self._transform(img)
        return img, label, idx

    def __len__(self):
        return len(self._label)

class MergeDataset(Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)


def merge_datasets(d0, d1):
    return MergeDataset(d0, d1)


if __name__=='__main__':

    set_logger()

    generate_val_indices = False
    if generate_val_indices:
        generate_val_indices_set()

    transform_train = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    new_class = np.arange(100).tolist()
    cub200_inst = CUB200(logger=logging, dataset='test', transform=transform_train, fix_class=new_class,
                         base_class=0, k_shot=10000, c_way=100)

    cub200_inst_ind = CUB200_Indexed(logger=logging, dataset='test', transform=transform_train, session=2)

    train_loader = DataLoader(cub200_inst, batch_size=4, num_workers=0, drop_last=False, shuffle=True)
    sum_exp = 0
    for k, batch in enumerate(train_loader):
        train_data, clf_labels, idx = batch

    print(sum_exp)