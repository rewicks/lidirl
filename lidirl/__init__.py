#!/usr/bin/env python
# -*- coding: utf-8 -*-


__version__ = '0.0.1'
__description__ = 'LID toolkit to improve performance on spontaneous noisy text with data augmentation.'

class DummyArgs():
    def __init__(self):
        return

def label(model="default",
            preprocess_path=None,
            input=None,
            output=None,
            complete=False,
            cpu=False,
            batch_size=500):
        from .label import label_langs as lidirl
        args = DummyArgs()
        args.model = model
        args.input = input
        args.output = output
        args.batch_size = batch_size
        args.complete = complete
        args.cpu = cpu
        args.batch_size = batch_size
        return lidirl(args)

def train():
    raise NotImplementedError

