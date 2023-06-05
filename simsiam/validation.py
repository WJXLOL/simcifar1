# https://github.com/zhirongw/lemniscate.pytorch/blob/master/test.py
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from torch import nn
from imbalance_data import cifar10Imbanlance, cifar100Imbanlance, dataset_lt_data
from simsiam.loader import TwoCropsTransform
from utils import util
from utils.util import *
import os


# KNN分类器
class KNNValidation(object):
    def __init__(self, args, model, K=1):
        self.model = model
        self.device = torch.device('cuda' if next(model.parameters()).is_cuda else 'cpu')
        self.args = args
        self.K = K

        base_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_train, transform_val = util.get_transform(args.dataset)
        if args.dataset == 'cifar10':
            train_set = cifar10Imbanlance.Cifar10Imbanlance(transform=base_transforms,
                                                            imbanlance_rate=args.imbanlance_rate, train=True,
                                                            file_path=args.data_root)

            # train_set.data = train_set.data['x']

            test_set = cifar10Imbanlance.Cifar10Imbanlance(imbanlance_rate=args.imbanlance_rate, train=False,
                                                           transform=base_transforms, file_path=args.data_root)

            # test_set.data = test_set.data['x']
            print("load cifar10")

        if args.dataset == 'cifar100':
            train_set = cifar100Imbanlance.Cifar100Imbanlance(transform=base_transforms,
                                                              imbanlance_rate=args.imbanlance_rate, train=True,
                                                              file_path=os.path.join(args.data_root,
                                                                                     'cifar-100-python/'))
            test_set = cifar100Imbanlance.Cifar100Imbanlance(imbanlance_rate=args.imbanlance_rate, train=False,
                                                             transform=base_transforms,
                                                             file_path=os.path.join(args.data_root,
                                                                                    'cifar-100-python/'))
            print("load cifar100")

        train_sampler = None
        self.train_dataloader = DataLoader(train_set, batch_size=args.batch_size,
                                           shuffle=False, num_workers=args.num_workers,
                                           persistent_workers=True, pin_memory=True,
                                           sampler=train_sampler,
                                           drop_last=True)
        self.val_dataloader = DataLoader(test_set, batch_size=args.batch_size,
                                         shuffle=False, num_workers=args.num_workers,
                                         persistent_workers=True,
                                         pin_memory=True,
                                         drop_last=True)

    def _topk_retrieval(self):
        """Extract features from validation split and search on train split features.

        result = self.train_dataloader.dataset.data.items()
        # Convert object to a list
        data = list(result)
        # Convert list to an array
        numpyArray = np.array(data)
        print(type(numpyArray))
        n_data = numpyArray.shape[0]
"""

        n_data = self.train_dataloader.dataset.x.shape[0]
        feat_dim = self.args.feat_dim

        self.model.eval()
        if str(self.device) == 'cuda':
            torch.cuda.empty_cache()

        train_features = torch.zeros([feat_dim, n_data], device=self.device)
        with torch.no_grad():
            for batch_idx, (inputs, _) in enumerate(self.train_dataloader):
                inputs = inputs.to(self.device)
                batch_size = inputs.size(0)

                # forward
                features = self.model(inputs)
                features = nn.functional.normalize(features)
                train_features[:, batch_idx * batch_size:batch_idx * batch_size + batch_size] = features.data.t()

            train_labels = torch.LongTensor(self.train_dataloader.dataset.targets).cuda()

        total = 0
        correct = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.val_dataloader):
                targets = targets.cuda(non_blocking=True)
                batch_size = inputs.size(0)
                features = self.model(inputs.to(self.device))

                dist = torch.mm(features, train_features)
                yd, yi = dist.topk(self.K, dim=1, largest=True, sorted=True)
                candidates = train_labels.view(1, -1).expand(batch_size, -1)
                retrieval = torch.gather(candidates, 1, yi)

                retrieval = retrieval.narrow(1, 0, 1).clone().view(-1)

                total += targets.size(0)
                correct += retrieval.eq(targets.data).sum().item()
        top1 = correct / total

        return top1

    def eval(self):
        return self._topk_retrieval()
