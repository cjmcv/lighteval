# MIT License

# Copyright (c) 2024 Taratra D. RAHARISON and The HuggingFace Team

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import logging
import logging.config

import colorlog
import typer

import lighteval.main_tasks
import lighteval.main_vllm


app = typer.Typer()

logging_config = dict(  # noqa C408
    version=1,
    formatters={
        "c": {
            "()": colorlog.ColoredFormatter,
            "format": "[%(asctime)s] [%(log_color)s%(levelname)8s%(reset)s]: %(message)s (%(filename)s:%(lineno)s)",
            "log_colors": {
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red,bg_white",
            },
        },
    },
    handlers={"h": {"class": "logging.StreamHandler", "formatter": "c", "level": logging.INFO}},
    root={
        "handlers": ["h"],
        "level": logging.INFO,
    },
)

logging.config.dictConfig(logging_config)
logging.captureWarnings(capture=True)

app.command(rich_help_panel="Evaluation Backends")(lighteval.main_vllm.vllm)
app.add_typer(
    lighteval.main_tasks.app,
    name="tasks",
    rich_help_panel="Utils",
    help="List or inspect tasks.",
)

if __name__ == "__main__":
    app()

# .vscode/launch.json
#
# {
#     // Use IntelliSense to learn about possible attributes.
#     // Hover to view descriptions of existing attributes.
#     // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
#     "version": "0.2.0",
#     "configurations": [
#         {
#             "name": "Python Debugger: Current File",
#             "type": "debugpy",
#             "request": "launch",
#             "program": "src/lighteval/__main__.py",
#             "console": "integratedTerminal",
#             "python": "/home/cjmcv/anaconda3/envs/eval-venv/bin/python",
#             "args": [
#                 "vllm",
#                 "pretrained=/home/cjmcv/project/llm_models/Qwen/Qwen2___5-1___5B-Instruct-AWQ,dtype=float16",
#                 "leaderboard|mmlu:abstract_algebra|0|0"
#             ]

#             //// 查看某个任务的具体内容，后面的|0|0 表示
#             // "args": [
#             //     "tasks",
#             //     "inspect",
#             //     "leaderboard|mmlu:abstract_algebra|0|0"
#             // ]

#             //// 查看任务列表
#             // "args": [
#             //     "tasks",
#             //     "list"
#             // ]
#         }
#     ]
# }