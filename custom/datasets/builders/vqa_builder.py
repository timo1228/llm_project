"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from custom.datasets.builders.base_dataset_builder import BaseDatasetBuilder

from custom.common.registry import registry
from custom.datasets.datasets.aok_vqa_datasets import AOKVQADataset, AOKVQAEvalDataset, AOKVQAInstructDataset
from custom.datasets.datasets.coco_vqa_datasets import COCOVQADataset, COCOVQAEvalDataset, COCOVQAInstructDataset
from custom.datasets.datasets.ocr_datasets import OCRVQADataset, OCRVQAInstructDataset

@registry.register_builder("coco_vqa")
class COCOVQABuilder(BaseDatasetBuilder):
    train_dataset_cls = COCOVQADataset
    eval_dataset_cls = COCOVQAEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/coco/defaults_vqa.yaml",
        "eval": "configs/datasets/coco/eval_vqa.yaml",
    }

@registry.register_builder("coco_vqa_instruct")
class COCOVQAInstructBuilder(BaseDatasetBuilder):
    train_dataset_cls = COCOVQAInstructDataset
    eval_dataset_cls = COCOVQAEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/coco/defaults_vqa_instruct.yaml",
        "eval": "configs/datasets/coco/eval_vqa.yaml",
    }

@registry.register_builder("ok_vqa")
class OKVQABuilder(COCOVQABuilder):
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/okvqa/defaults.yaml",
    }

@registry.register_builder("ok_vqa_instruct")
class OKVQAInstructBuilder(COCOVQAInstructBuilder):
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/okvqa/defaults_instruct.yaml",
    }

@registry.register_builder("aok_vqa")
class AOKVQABuilder(BaseDatasetBuilder):
    train_dataset_cls = AOKVQADataset
    eval_dataset_cls = AOKVQAEvalDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/aokvqa/defaults.yaml"}

@registry.register_builder("aok_vqa_instruct")
class AOKVQAInstructBuilder(BaseDatasetBuilder):
    train_dataset_cls = AOKVQAInstructDataset
    eval_dataset_cls = AOKVQAEvalDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/aokvqa/defaults_instruct.yaml"}


@registry.register_builder("ocr_vqa")
class OCRVQABuilder(BaseDatasetBuilder):
    train_dataset_cls = OCRVQADataset
    eval_dataset_cls = OCRVQADataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/ocrvqa/defaults.yaml"}

@registry.register_builder("ocr_vqa_instruct")
class OCRVQAInstructBuilder(BaseDatasetBuilder):
    train_dataset_cls = OCRVQAInstructDataset
    eval_dataset_cls = OCRVQAInstructDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/ocrvqa/defaults_instruct.yaml"}


