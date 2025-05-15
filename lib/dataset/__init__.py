# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lib.dataset.panoptic import Panoptic as panoptic
from lib.dataset.panoptic_pred import PanopticInference as panoptic_pred
from lib.dataset.shelf_synthetic import ShelfSynthetic as shelf_synthetic
from lib.dataset.campus_synthetic import CampusSynthetic as campus_synthetic
from lib.dataset.shelf import Shelf as shelf
from lib.dataset.campus import Campus as campus


def get_dataset(dataset):
    if dataset == 'panoptic':
        return panoptic
    elif dataset == 'panoptic_pred':
        return panoptic_pred
    elif dataset == 'shelf_synthetic':
        return shelf_synthetic
    elif dataset == 'campus_synthetic':
        return campus_synthetic
    elif dataset == 'shelf':
        return shelf
    elif dataset == 'campus':
        return campus
    else:
        raise ValueError('Unknown dataset')

