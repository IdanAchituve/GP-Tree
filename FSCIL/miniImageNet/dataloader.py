from pathlib import Path
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

"""
Mini-imagenet dataset for learning with episodes
Data available here: https://drive.google.com/u/0/uc?id=0B3Irx3uQNoBMQ1FlNXJsZUdYWEE
Episode index list available here: https://github.com/xyutao/fscil/tree/master/data/index_list/mini_imagenet
"""


def class_map(path):
    all_clsses = []
    for i in range(1, 10):
        curr_file = open(Path(path) / "index_list" / f"session_{i}.txt", 'r')
        lines = [l for l in curr_file]
        curr_lbls = list(set([l.split('/')[-2] for l in lines]))
        all_clsses += curr_lbls

    return {k: v for v, k in enumerate(all_clsses)}


class EpisodeDataset(Dataset):
    def __init__(self, path, episode_id, class_map, split='train', use_val=False, transform=None):
        assert 1 <= episode_id <= 9
        assert split in ('train', 'val', 'test')
        if episode_id > 1:
            assert split != 'val', 'we have val split for episode 1 only!'
        if split == 'val':
            assert use_val, "set use_val=True for val split"

        self.split = split
        self.path = Path(path)
        self.episode_id = episode_id
        self.transform = transforms.Compose([transforms.ToTensor()]) if transform is None else transform
        self.class_map = class_map

        if self.split == 'train':
            if use_val and (episode_id == 1):
                idx_file = open(Path(path) / "index_list" / f"session_{episode_id}_train.txt", 'r')
                lines = [l for l in idx_file]
            else:
                idx_file = open(Path(path) / "index_list" / f"session_{episode_id}.txt", 'r')
                lines = [l for l in idx_file]
        elif self.split == 'val':
            # NOTE: if we are here then episode_id == 1
            idx_file = open(Path(path) / "index_list" / f"session_{episode_id}_val.txt", 'r')
            lines = [l for l in idx_file]
        else:
            # NOTE: test split, also taking all previous episodes
            lines = []
            for i in range(1, episode_id + 1):
                idx_file = open(Path(path) / "index_list" / f"test_{i}.txt", 'r')
                lines += [l for l in idx_file]

        self.images_name = [Path(l).name.rstrip() for l in lines]
        self.targets = [l.split('/')[-2] for l in lines]

    def __len__(self):
        return len(self.images_name)

    def __getitem__(self, item):
        image_path = self.path / "images" / self.images_name[item]
        image = Image.open(image_path).convert('RGB')

        image = self.transform(image)

        return image, torch.tensor(self.class_map[self.targets[item]], dtype=torch.long), item


class MiniImagenetEpisodes:

    def __init__(self, root, transform_train=None, transform_test=None, use_val=False):
        """
        :param root: path to data root dir
        :param transform_train:
        :param transform_test:
        :param use_val: bool. If True using val split taken from session 1
        """
        self.root = root
        self.use_val = use_val
        self.class_map = class_map(root)

        if transform_train is None:
            transform_train = transforms.Compose([
                transforms.Resize((84, 84)),
                transforms.RandomCrop(84, padding=8),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

            transform_test = transforms.Compose([
                transforms.Resize((84, 84)),
                transforms.CenterCrop(84),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

        self.transform_train = transform_train
        self.transform_test = transform_test

    def get_episode_datasets(self, episode_id):
        if self.use_val and (episode_id == 1):
            train_set = EpisodeDataset(
                path=self.root, episode_id=episode_id, split='train', transform=self.transform_train,
                class_map=self.class_map, use_val=self.use_val
            )
            test_set = EpisodeDataset(
                path=self.root, episode_id=episode_id, split='test', transform=self.transform_test,
                class_map=self.class_map, use_val=self.use_val
            )
            val_set = EpisodeDataset(
                path=self.root, episode_id=episode_id, split='val', transform=self.transform_test,
                class_map=self.class_map, use_val=self.use_val
            )

            return train_set, val_set, test_set
        else:
            train_set = EpisodeDataset(
                path=self.root, episode_id=episode_id, split='train', transform=self.transform_train,
                class_map=self.class_map, use_val=self.use_val
            )
            test_set = EpisodeDataset(
                path=self.root, episode_id=episode_id, split='test', transform=self.transform_test,
                class_map=self.class_map, use_val=self.use_val
            )
            return train_set, test_set


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # no val
    data = MiniImagenetEpisodes(root='data')
    for episode_id in range(1, 10):
        episode_train, episode_test = data.get_episode_datasets(episode_id)
        print(f"Episede {episode_id}, train size {len(episode_train)}, test size {len(episode_test)}")

    # no with val
    data = MiniImagenetEpisodes(root='data', use_val=True)
    for episode_id in range(1, 10):
        if episode_id == 1:
            episode_train, episode_val, episode_test = data.get_episode_datasets(episode_id)
            print(
                f"Episede {episode_id}, train size {len(episode_train)}, val size {len(episode_val)}, "
                f"test size {len(episode_test)}"
            )
        else:
            episode_train, episode_test = data.get_episode_datasets(episode_id)
            print(f"Episede {episode_id}, train size {len(episode_train)}, val size 0, test size {len(episode_test)}")

    # loader
    train_loader = torch.utils.data.DataLoader(episode_train, batch_size=128)
    batch = next(iter(train_loader))
    x, y, _ = batch

    # sample image
    plt.imshow(episode_train.__getitem__(4)[0].permute(1, 2, 0).numpy())
    plt.show()