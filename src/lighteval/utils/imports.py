# Copyright 2024 The HuggingFace Team. All rights reserved.
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

import importlib


def is_vllm_available() -> bool:
    return importlib.util.find_spec("vllm") is not None

NO_VLLM_ERROR_MSG = "You are trying to use an VLLM model, for which you need `vllm`, which is not available in your environment. Please install it using pip, `pip install vllm`."

def can_load_spacy_tokenizer(language: str) -> bool:
    imports = []
    packages = ["spacy", "stanza"]
    if language == "vi":
        packages.append("pyvi")
    elif language == "zh":
        packages.append("jieba")

    for package in packages:
        imports.append(importlib.util.find_spec(package))
    return all(cur_import is not None for cur_import in imports)


NO_SPACY_TOKENIZER_ERROR_MSG = "You are trying to load a spacy tokenizer, for which you need `spacy` and its dependencies, which are not available in your environment. Please install them using `pip install lighteval[multilingual]`."


def can_load_stanza_tokenizer() -> bool:
    return importlib.util.find_spec("stanza") is not None


NO_STANZA_TOKENIZER_ERROR_MSG = "You are trying to load a stanza tokenizer, for which you need `stanza`, which is not available in your environment. Please install it using `pip install lighteval[multilingual]`."
