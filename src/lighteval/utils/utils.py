# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from dataclasses import asdict, dataclass, is_dataclass
from typing import Callable, TypeVar, Tuple

import numpy as np
from datasets import DatasetDict, load_dataset, DownloadConfig, load_from_disk, get_dataset_config_info
from pytablewriter import MarkdownTableWriter


def flatten_dict(nested: dict, sep="/") -> dict:
    """Flatten dictionary, list, tuple and concatenate nested keys with separator."""

    def clean_markdown(v: str) -> str:
        return v.replace("|", "_").replace("\n", "_") if isinstance(v, str) else v  # Need this for markdown

    def rec(nest: dict, prefix: str, into: dict):
        for k, v in sorted(nest.items()):
            # if sep in k:
            #     raise ValueError(f"separator '{sep}' not allowed to be in key '{k}'")
            if isinstance(v, dict):
                rec(v, prefix + k + sep, into)
            elif isinstance(v, (list, tuple)):
                for i, vv in enumerate(v):
                    if isinstance(vv, dict):
                        rec(vv, prefix + k + sep + str(i) + sep, into)
                    else:
                        vv = (
                            vv.replace("|", "_").replace("\n", "_") if isinstance(vv, str) else vv
                        )  # Need this for markdown
                        into[prefix + k + sep + str(i)] = vv.tolist() if isinstance(vv, np.ndarray) else vv
            elif isinstance(v, np.ndarray):
                into[prefix + k + sep + str(i)] = v.tolist()
            else:
                v = clean_markdown(v)
                into[prefix + k] = v

    flat = {}
    rec(nested, "", flat)
    return flat


def clean_s3_links(value: str) -> str:
    """Cleans and formats s3 bucket links for better display in the result table (nanotron models)

    Args:
        value (str): path to clean

    Returns:
        str : cleaned path
    """
    s3_bucket, s3_prefix = str(value).replace("s3://", "").split("/", maxsplit=1)
    if not s3_prefix.endswith("/"):
        s3_prefix += "/"
    link_str = f"https://s3.console.aws.amazon.com/s3/buckets/{s3_bucket}?prefix={s3_prefix}"
    value = f'<a href="{link_str}" target="_blank"> {value} </a>'
    return value

def sanitize_numpy(example_dict: dict) -> dict:
    """
    Sanitizes a dictionary by converting any numpy generic types to their corresponding Python types.

    Args:
        example_dict (dict): The dictionary to be sanitized.

    Returns:
        dict: The sanitized dictionary with numpy generic types converted to Python types.
    """
    output_dict = {}
    for k, v in example_dict.items():
        if isinstance(v, np.generic):
            output_dict[k] = v.item()
        else:
            output_dict[k] = v
    return output_dict


ListLikeTypeVar = TypeVar("ListLikeTypeVar")
ListLike = list[ListLikeTypeVar] | tuple[ListLikeTypeVar, ...]


ElementType = TypeVar("ElementType")


def as_list(item: ListLike[ElementType] | ElementType) -> list[ElementType]:
    """
    Convert the given item into a list.

    If the item is already a list, it is returned as is.
    If the item is a tuple, it is converted into a list.
    Otherwise, the item is wrapped in a list.

    Args:
        item (Union[list, tuple, Any]): The item to be converted.

    Returns:
        list: The converted list.

    """
    if isinstance(item, list):
        return item

    elif isinstance(item, tuple):
        return list(item)

    return [item]


def make_results_table(result_dict):
    """Generate table of results."""
    md_writer = MarkdownTableWriter()
    md_writer.headers = ["Task", "Version", "Metric", "Value", "", "Stderr"]

    values = []

    for k in sorted(result_dict["results"].keys()):
        dic = result_dict["results"][k]
        version = result_dict["versions"][k] if k in result_dict["versions"] else ""
        for m, v in dic.items():
            if m.endswith("_stderr"):
                continue

            if m + "_stderr" in dic:
                se = dic[m + "_stderr"]
                values.append([k, version, m, "%.4f" % v, "±", "%.4f" % se])
            else:
                values.append([k, version, m, "%.4f" % v, "", ""])
            k = ""
            version = ""
    md_writer.value_matrix = values

    return md_writer.dumps()


@dataclass
class EnvConfig:
    """
    Configuration class for environment settings.

    Attributes:
        cache_dir (str): directory for caching data.
        token (str): authentication token used for accessing the HuggingFace Hub.
    """

    cache_dir: str = os.getenv("HF_HUB_CACHE", "/scratch")
    token: str = os.getenv("HF_TOKEN")

def download_dataset_worker(
    dataset_path: str,
    dataset_config_name: str,
    trust_dataset: bool,
    dataset_filter: Callable[[dict], bool] | None = None,
    revision: str | None = None,
    splits: list[str] = [],
) -> Tuple[DatasetDict, bool]:
    """
    Worker function to download a dataset from the HuggingFace Hub.
    Used for parallel dataset loading.
    """
    # 先尝试读取本地数据
    is_local_dataset = True
    if is_local_dataset:
        dataset_cache_dir = os.environ.get('HF_DATASETS_CACHE')+'/'+ dataset_path.replace("/", "___") + "/" + dataset_config_name
        if os.path.exists(dataset_cache_dir):
            # Skip the next level of the path
            contents = os.listdir(dataset_cache_dir)
            sub_folder = contents[0]
            dataset_cache_dir = os.path.join(dataset_cache_dir, sub_folder, "")
            # Get the lastest version 
            versions = [os.path.join(dataset_cache_dir, d) for d in os.listdir(dataset_cache_dir) if os.path.isdir(os.path.join(dataset_cache_dir, d))]
            latest_version_data_set = max(versions, key=os.path.getmtime) if versions else None
            # Get all target arrows with splits.
            arrow_list = os.listdir(latest_version_data_set)
            # datasets = DatasetDict
            dataset = []
            for s in splits:
                for item in arrow_list:
                    if s in item:
                        latest_version_test_dataset = os.path.join(latest_version_data_set, item, "")
                        # 单独加载arrow都只有一个'train', 无论其本身是test/validation/train。而从网上加载，则会用字段区分。
                        # 所以从本地单独加载，需在这里先过滤split(即train/test/validation字段)选择对应arrow文件后加载，从网上加载则先加载整体到dataset后，再过滤split。
                        dataset.append(load_dataset('arrow', data_files=latest_version_test_dataset)['train'])

            print("len(dataset)", len(dataset))
            if len(dataset) == 0:
                is_local_dataset = False
                raise Warning(f" Unable to load data from the local location. Try to obtain it from the network..")
        else:
            is_local_dataset = False
            print(f"dataset_cache_dir {dataset_cache_dir} is not exist. Try to obtain it from the network..")

    if not is_local_dataset:
        dataset = load_dataset(
            path=dataset_path, # "/home/cjmcv/project/llm_datasets/huggingface/lighteval___mmlu/abstract_algebra/1.0.0/e24764f1fb58c26b5f622157644f2e5fe77e5b01", # dataset_path, # 'lighteval/mmlu' # https://huggingface.co/datasets/lighteval/mmlu
            name=dataset_config_name, #dataset_config_name, # 'abstract_algebra'
            data_dir=None, # "/home/cjmcv/project/llm_datasets/huggingface/lighteval___mmlu/abstract_algebra/1.0.0/1789618b211cec2e9545c1a41c62b7c6b2b2ccc0dbb64b7e3c89867b0a538891",
            cache_dir=None, # os.environ.get('HF_DATASETS_CACHE'),   # os.environ.get('HF_DATASETS_CACHE')
            download_mode=None,
            trust_remote_code=trust_dataset,
            revision=revision,
        )

    if dataset_filter is not None:
        dataset = dataset.filter(dataset_filter)  # 如果是加载离线数据，则这里不能调用

    # It returns DatasetDict because we don't specify a split
    return dataset, is_local_dataset  # type: ignore


def safe_divide(numerator: np.ndarray, denominator: float, default_value: float = 0.0) -> np.ndarray:
    return np.where(denominator != 0, numerator / denominator, default_value)
