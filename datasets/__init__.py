from .mvtec import MVTEC
from .mvtec import MVTECANO, get_normal_image_paths_mvtec
from .btad import BTAD, get_normal_image_paths_btad
from .mvtec_3d import MVTEC3D, get_normal_image_paths_mvtec3d
from .mvtec_loco import MVTECLOCO
from .visa import VISA, get_normal_image_paths_visa
import torch
from .mvtec_bgad import MVTEC_CLASS_NAMES, MVTecDataset, MVTecFSCopyPasteDataset, MVTecFSDataset, MVTecPseudoDataset, MVTecAnomalyDataset
from .utils_bgad import BalancedBatchSampler

__all__ = ['MVTEC', 'MVTECANO', 'BTAD', 'MVTEC3D', 'MVTECLOCO', 'VISA']


def get_normal_image_paths(root, class_name, dataset='mvtec'):
    if dataset == 'mvtec':
        return get_normal_image_paths_mvtec(root, class_name)
    elif dataset == 'btad':
        return get_normal_image_paths_btad(root, class_name)
    elif dataset == 'visa':
        return get_normal_image_paths_visa(root, class_name)
    elif dataset == 'mvtec3d':
        return get_normal_image_paths_mvtec3d(root, class_name)
    else:
        raise ValueError("Unrecognized!")
def create_data_loader(args):
    kwargs = {'num_workers': args.num_workers, 'pin_memory': True}
    if args.dataset == 'mvtec':
        train_dataset = MVTecDataset(args, is_train=True)
        test_dataset  = MVTecDataset(args, is_train=False)
    elif args.dataset == 'btad':
        train_dataset = BTADDataset(args, is_train=True)
        test_dataset  = BTADDataset(args, is_train=False)
    else:
        raise NotImplementedError('{} is not supported dataset!'.format(args.dataset))
    # dataloader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False, **kwargs)

    return train_loader, test_loader


def create_fas_data_loader(args):
    kwargs = {'num_workers': args.num_workers, 'pin_memory': True}
    if args.dataset == 'mvtec':
        normal_dataset = MVTecDataset(args, is_train=True)
        if args.data_strategy == '0':
            train_dataset = MVTecFSDataset(args, is_train=True)
        elif args.data_strategy == '0,1':
            train_dataset = MVTecFSCopyPasteDataset(args, is_train=True)
        elif args.data_strategy == '0,2':
            train_dataset = MVTecPseudoDataset(args, is_train=True)
        elif args.data_strategy == '0,1,2':
            train_dataset = MVTecAnomalyDataset(args, is_train=True)
        if args.not_in_test:
            test_dataset  = MVTecDataset(args, is_train=False, excluded_images=train_dataset.a_imgs)
        else:
            test_dataset  = MVTecDataset(args, is_train=False)
    elif args.dataset == 'btad':
        normal_dataset = BTADDataset(args, is_train=True)
        if args.data_strategy == '0':
            train_dataset = BTADFSDataset(args, is_train=True)
        elif args.data_strategy == '0,1':
            train_dataset = BTADFSCopyPasteDataset(args, is_train=True)
        if args.not_in_test:
            test_dataset  = BTADDataset(args, is_train=False, excluded_images=train_dataset.a_imgs)
        else:
            test_dataset  = BTADDataset(args, is_train=False)
    else:
        raise NotImplementedError('{} is not supported dataset!'.format(args.dataset))
    # dataloader
    normal_loader = torch.utils.data.DataLoader(normal_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs)
    if args.balanced_data_loader:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=BalancedBatchSampler(args, train_dataset), **kwargs)
    else:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False, **kwargs)

    return normal_loader, train_loader, test_loader


def create_test_data_loader(args):
    kwargs = {'num_workers': args.num_workers, 'pin_memory': True}
    if args.dataset == 'mvtec':
        test_dataset  = MVTecDataset(args, is_train=False)
    elif args.dataset == 'btad':
        test_dataset  = BTADDataset(args, is_train=False)
    else:
        raise NotImplementedError('{} is not supported dataset!'.format(args.dataset))
    # dataloader
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False, **kwargs)

    return test_loader
