#!/usr/bin/env python
# -*-coding:utf-8-*-


"""
"""


import torchvision
#
from deepobs import config
from deepobs.pytorch.datasets import cifar10, cifar100, mnist
from deepobs.pytorch.datasets.cifar10 import training_transform_not_augmented

#
from .samplers import SeededRandomSampler, SubsetSampler


# ##############################################################################
# # RESAMPLED MNIST
# ##############################################################################
class mnist_resampled(mnist):
    """
    Like the superclass, but the dataloader adds a transform to reduce size.
    """

    SIZE = None

    def __init__(self, batch_size, train_eval_size=10000, random_seed=0):
        """ """
        self.random_seed = random_seed
        super().__init__(batch_size, train_eval_size)

    def get_transform(self):
        """ """
        if self.SIZE is None:
            t = torchvision.transforms.ToTensor()
        else:
            t = torchvision.transforms.Compose(
                [
                    torchvision.transforms.Resize(self.SIZE),
                    torchvision.transforms.ToTensor(),
                ]
            )
        return t

    def _make_test_dataloader(self):
        """ """
        transform = self.get_transform()
        ds = torchvision.datasets.MNIST(
            root=config.get_data_dir(),
            train=False,
            download=True,
            transform=transform,
        )
        dl = self._make_dataloader(ds, sampler=None, shuffle=False)
        return dl

    def _make_train_eval_dataloader(self):
        """ """
        # Replace the truncated, implicit-seed sampler with our balanced,
        # reproducible one. Otherwise, evaluating at different times messes the
        # seed and also kills reproducibility during training.
        transform = self.get_transform()
        ds = torchvision.datasets.MNIST(
            root=config.get_data_dir(),
            train=True,
            download=True,
            transform=transform,
        )
        #
        sampler = SubsetSampler.get_balanced(
            self._train_dataloader.dataset,
            size=self._train_eval_size,
            random=False,
        )
        #
        dl = self._make_dataloader(ds, sampler=sampler, shuffle=False)
        return dl

    def _make_train_and_valid_dataloader(self):
        """ """
        transform = self.get_transform()
        ds = torchvision.datasets.MNIST(
            root=config.get_data_dir(),
            train=True,
            download=True,
            transform=transform,
        )
        #
        # given a seed, split into training and xv subsets.
        # note that self._train_eval_size is the size of the val, not the
        # train set (maybe was a bug in OBS?).
        indices = list(SeededRandomSampler.randperm(len(ds), self.random_seed))
        valid_indices, train_indices = (
            indices[: self._train_eval_size],
            indices[self._train_eval_size :],
        )
        #
        train_sampler = SeededRandomSampler(train_indices, self.random_seed)
        valid_sampler = SeededRandomSampler(valid_indices, self.random_seed)
        # since random sampling, shuffle is useless
        train_loader = self._make_dataloader(ds, sampler=train_sampler)
        valid_loader = self._make_dataloader(ds, sampler=valid_sampler)
        return train_loader, valid_loader


class mnist_16x16(mnist_resampled):
    """ """

    SIZE = 16


# ##############################################################################
# # CIFAR 10 WITHOUT TRAINING DATA AUGMENTATION
# ##############################################################################
class cifar10det(cifar10):
    """
    Deterministic: no training data augmentation.
    Plus all the reproducible sampler extensions.
    """

    def __init__(
        self,
        batch_size,
        data_augmentation=False,
        train_eval_size=10000,
        random_seed=0,
    ):
        """ """
        self.random_seed = random_seed
        super().__init__(batch_size, False, train_eval_size)

    def _make_test_dataloader(self):
        transform = training_transform_not_augmented
        test_dataset = torchvision.datasets.CIFAR10(
            root=config.get_data_dir(),
            train=False,
            download=True,
            transform=transform,
        )
        return self._make_dataloader(test_dataset, sampler=None, shuffle=False)

    def _make_train_eval_dataloader(self):
        """ """
        ds = self._train_dataloader.dataset
        sampler = SubsetSampler.get_balanced(
            ds, size=self._train_eval_size, random=False
        )
        dl = self._make_dataloader(ds, sampler=sampler, shuffle=False)
        return dl

    def _make_train_and_valid_dataloader(self):
        """ """
        # always not augmented
        transform = training_transform_not_augmented

        ds = torchvision.datasets.CIFAR10(
            root=config.get_data_dir(),
            train=True,
            download=True,
            transform=transform,
        )
        # given a seed, split into training and xv subsets.
        # note that self._train_eval_size is the size of the val, not the
        # train set (maybe was a bug in OBS?).
        indices = list(SeededRandomSampler.randperm(len(ds), self.random_seed))
        valid_indices, train_indices = (
            indices[: self._train_eval_size],
            indices[self._train_eval_size :],
        )
        #
        train_sampler = SeededRandomSampler(train_indices, self.random_seed)
        valid_sampler = SeededRandomSampler(valid_indices, self.random_seed)
        # since random sampling, shuffle is useless
        train_loader = self._make_dataloader(ds, sampler=train_sampler)
        valid_loader = self._make_dataloader(ds, sampler=valid_sampler)
        return train_loader, valid_loader


# ##############################################################################
# # CIFAR 100 WITH DETERMINISTIC AUGMENTATIONS
# ##############################################################################
class cifar100det(cifar100):
    """
    Deterministic: no training data augmentation
    """

    def __init__(
        self,
        batch_size,
        data_augmentation=True,
        train_eval_size=10000,
        random_seed=0,
    ):
        """ """
        self.random_seed = random_seed
        super().__init__(batch_size, False, train_eval_size)

    def _make_test_dataloader(self):
        transform = training_transform_not_augmented
        test_dataset = torchvision.datasets.CIFAR100(
            root=config.get_data_dir(),
            train=False,
            download=True,
            transform=transform,
        )
        return self._make_dataloader(test_dataset, sampler=None, shuffle=False)

    def _make_train_eval_dataloader(self):
        """ """
        ds = self._train_dataloader.dataset
        sampler = SubsetSampler.get_balanced(
            ds, size=self._train_eval_size, random=False
        )
        dl = self._make_dataloader(ds, sampler=sampler, shuffle=False)
        return dl

    def _make_train_and_valid_dataloader(self):
        """ """
        # always not augmented
        transform = training_transform_not_augmented

        ds = torchvision.datasets.CIFAR100(
            root=config.get_data_dir(),
            train=True,
            download=True,
            transform=transform,
        )
        # given a seed, split into training and xv subsets.
        # note that self._train_eval_size is the size of the val, not the
        # train set (maybe was a bug in OBS?).
        indices = list(SeededRandomSampler.randperm(len(ds), self.random_seed))
        valid_indices, train_indices = (
            indices[: self._train_eval_size],
            indices[self._train_eval_size :],
        )
        #
        train_sampler = SeededRandomSampler(train_indices, self.random_seed)
        valid_sampler = SeededRandomSampler(valid_indices, self.random_seed)
        # since random sampling, shuffle is useless
        train_loader = self._make_dataloader(ds, sampler=train_sampler)
        valid_loader = self._make_dataloader(ds, sampler=valid_sampler)
        return train_loader, valid_loader
