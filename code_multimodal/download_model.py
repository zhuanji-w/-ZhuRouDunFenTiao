#!/usr/bin/env python3
"""
简单的 Hugging Face 模型下载脚本。

功能：
- 输入模型名称（如：stabilityai/stable-diffusion-2-1），通过 huggingface-cli
  将该模型的所有文件（包含子目录）完整下载到本地。

依赖：
- pip install "huggingface_hub[cli]"
- 需要命令行中可用的 `huggingface-cli`（安装 huggingface_hub 后自带）

用法示例：
- 交互式：python download_model.py
- 命令行参数：
    python download_model.py stabilityai/stable-diffusion-2-1 \\
        --local-dir ./model/text2img
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def ensure_cli_available() -> None:
    """检查 huggingface-cli 是否可用，不可用时给出提示。"""
    if shutil.which("huggingface-cli") is None:
        print(
            "未找到 `huggingface-cli` 命令，请先安装：\n"
            "  pip install \"huggingface_hub[cli]\"\n",
            file=sys.stderr,
        )
        sys.exit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="通过 huggingface-cli 下载 Hugging Face 模型所有文件。")
    parser.add_argument(
        "model_id",
        nargs="?",
        help="模型名称，例如：stabilityai/stable-diffusion-2-1",
    )
    parser.add_argument(
        "--local-dir",
        type=str,
        default=None,
        help="本地保存目录，默认：当前脚本所在目录下的 ./model/<model_id_last_part>",
    )
    parser.add_argument(
        "--repo-type",
        type=str,
        default="model",
        help="仓库类型，一般为 model，保持默认即可。",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="可选：指定分支/Tag/Commit，如 main、v1.0 等。",
    )
    return parser.parse_args()


def main() -> None:
    ensure_cli_available()
    args = parse_args()

    model_id = args.model_id
    if not model_id:
        model_id = input("请输入 Hugging Face 模型名称（如 stabilityai/stable-diffusion-2-1）：").strip()
        if not model_id:
            print("模型名称不能为空。")
            sys.exit(1)

    # 默认下载目录：脚本所在目录的 model/<最后一段>
    script_dir = Path(__file__).resolve().parent
    if args.local_dir:
        local_dir = Path(args.local_dir).expanduser().resolve()
    else:
        model_short = model_id.split("/")[-1]
        local_dir = (script_dir / "model" / model_short).resolve()

    local_dir.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "huggingface-cli",
        "download",
        model_id,
        "--repo-type",
        args.repo_type,
        "--local-dir",
        str(local_dir),
        "--local-dir-use-symlinks",
        "False",
    ]
    if args.revision:
        cmd.extend(["--revision", args.revision])

    print("将执行命令：", " ".join(cmd))
    print(f"目标下载目录：{local_dir}")

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"\n[错误] huggingface-cli 下载失败，退出码：{exc.returncode}", file=sys.stderr)
        sys.exit(exc.returncode)

    print("\n下载完成！模型已保存到：", local_dir)


if __name__ == "__main__":
    main()

