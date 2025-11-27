"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
from custom.common.registry import registry

from custom.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from custom.datasets.datasets.image_text_pair_datasets import ImageTextPairDataset, ImageTextPairInstructDataset

@registry.register_builder("conceptual_caption_3m")
class ConceptualCaption3MBuilder(BaseDatasetBuilder):
    train_dataset_cls = ImageTextPairDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/conceptual_caption/defaults_3m.yaml"
    }

@registry.register_builder("conceptual_caption_3m_instruct")
class ConceptualCaption3MInstructBuilder(BaseDatasetBuilder):
    train_dataset_cls = ImageTextPairInstructDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/conceptual_caption/defaults_3m_instruct.yaml"
    }


@registry.register_builder("conceptual_caption_12m")
class ConceptualCaption12MBuilder(BaseDatasetBuilder):
    train_dataset_cls = ImageTextPairDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/conceptual_caption/defaults_12m.yaml"
    }

@registry.register_builder("conceptual_caption_12m_instruct")
class ConceptualCaption12MInstructBuilder(BaseDatasetBuilder):
    train_dataset_cls = ImageTextPairInstructDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/conceptual_caption/defaults_12m_instruct.yaml"
    }

@registry.register_builder("sbu_caption")
class SBUCaptionBuilder(BaseDatasetBuilder):
    train_dataset_cls = ImageTextPairDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/sbu_caption/defaults.yaml"}


@registry.register_builder("sbu_caption_instruct")
class SBUCaptionInstructBuilder(BaseDatasetBuilder):
    train_dataset_cls = ImageTextPairInstructDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/sbu_caption/defaults_instruct.yaml"}


@registry.register_builder("vg_caption")
class VGCaptionBuilder(BaseDatasetBuilder):
    train_dataset_cls = ImageTextPairDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/vg/defaults_caption.yaml"}


@registry.register_builder("vg_caption_instruct")
class VGCaptionInstructBuilder(BaseDatasetBuilder):
    train_dataset_cls = ImageTextPairInstructDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/vg/defaults_caption_instruct.yaml"}
