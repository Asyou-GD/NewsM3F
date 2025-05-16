#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compute two-level multi-label F1 for jsonl formatted like:
{"label": "label_level1:Politics, International\nlabel_level2:Parliament",
 "predict": "label_level1:Politics\nlabel_level2:Courts"}
"""

import json, re, ast
from pathlib import Path
from typing import List, Dict, Tuple, Any

import numpy as np
from sklearn.metrics import f1_score   # pip install scikit-learn>=1.0

from pathlib import Path
import json, ast
from typing import List, Dict, Any

def load_loose_jsonl(path: str) -> List[Dict[str, Any]]:
    """
    一行一条 JSON 记录，额外把 predict 中的 JSON 字符串解析成 dict。
    """
    samples: List[Dict[str, Any]] = []
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        for attempt in range(3):
            try:
                if attempt == 0:
                    obj = json.loads(line)
                elif attempt == 1:
                    obj = json.loads(line.replace("'", '"'))
                else:
                    obj = ast.literal_eval(line)
                break
            except Exception as e:
                last_err = e
        else:
            print(f"[WARN] 跳过无法解析的样本：{last_err}\n{line[:200]}…")
            continue

        p = obj.get("predict")
        if isinstance(p, str):
            p_strip = p.strip()
            if p_strip.startswith("{") and p_strip.endswith("}"):
                try:
                    obj["predict"] = json.loads(p_strip)
                except Exception:
                    try:
                        obj["predict"] = json.loads(p_strip.replace("'", '"'))
                    except Exception:
                        pass

        samples.append(obj)

    return samples


# ───────────────────────── 2. 标签解析/编码 ──────────────────────────
ALIAS = {                   # 拼写对齐的简易映射，可按需扩充
    "Political": "Politics",
    "LawCourt":  "Courts",
}

def extract_tags(text: Any, key: str) -> List[str]:
    """
    支持两种输入格式
    1) 字符串:  "label_level1:Sport,\nlabel_level2:Football (Soccer)\n"
    2) 字典:    {"label_level2": ["Football (Soccer)", ...]}
    """
    # ---------- dict ----------
    if isinstance(text, dict):
        val = text.get(key, [])
        if isinstance(val, list):
            return [v.strip() for v in val if v.strip()]
        if isinstance(val, str):
            return [val.strip()] if val.strip() else []
        return []

    # ---------- str ----------
    if not isinstance(text, str):
        return []

    m = re.search(rf"{key}\s*[:：]\s*([^\n\r]*)", text, flags=re.I)
    if not m:
        return []
    seg = m.group(1)
    seg = re.split(r"label_level\d\s*[:：]", seg, maxsplit=1)[0]

    # 只按逗号切分；保留空格与括号等字符
    parts = re.split(r"[,\uff0c]", seg)
    return [p.strip() for p in parts if p.strip()]


def build_class_index(samples: List[dict], level_key: str) -> Dict[str, int]:
    classes = set()
    for item in samples:
        classes.update(extract_tags(item["label"], level_key))
    return {cls: idx for idx, cls in enumerate(sorted(classes))}


def encode_multi_hot(tags: List[str], class2idx: Dict[str, int]) -> np.ndarray:
    vec = np.zeros(len(class2idx), dtype=int)
    for t in tags:
        t = ALIAS.get(t, t)
        if t in class2idx:
            vec[class2idx[t]] = 1
    return vec


def accumulate_arrays(
    samples: List[dict],
    class2idx_l1: Dict[str, int],
    class2idx_l2: Dict[str, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    y_true_l1, y_pred_l1 = [], []
    y_true_l2, y_pred_l2 = [], []

    for item in samples:
        gt_l1 = extract_tags(item["label"], "label_level1")
        gt_l2 = extract_tags(item["label"], "label_level2")

        pr_l1 = [ALIAS.get(t, t) for t in extract_tags(item.get("predict", ""), "label_level1")]
        pr_l2 = [ALIAS.get(t, t) for t in extract_tags(item.get("predict", ""), "label_level2")]

        pr_l1 = [t for t in pr_l1 if t in class2idx_l1]
        pr_l2 = [t for t in pr_l2 if t in class2idx_l2]
        y_true_l1.append(encode_multi_hot(gt_l1, class2idx_l1))
        y_pred_l1.append(encode_multi_hot(pr_l1, class2idx_l1))
        y_true_l2.append(encode_multi_hot(gt_l2, class2idx_l2))
        y_pred_l2.append(encode_multi_hot(pr_l2, class2idx_l2))


    if not y_true_l1:
        raise RuntimeError("0 个样本被成功加载，请检查文件路径或内容格式。")
    return (
        np.vstack(y_true_l1),
        np.vstack(y_pred_l1),
        np.vstack(y_true_l2),
        np.vstack(y_pred_l2),
    )


# ───────────────────────── 3. 主函数 ──────────────────────────
def compute_two_level_f1(jsonl_path: str) -> Dict[str, float]:
    """
    同时计算 Macro / Micro / Weighted 三种平均方式的 F1。
    返回键示例:
        f1_macro_l1, f1_micro_l1, f1_weighted_l1,
        f1_macro_l2, f1_micro_l2, f1_weighted_l2
    """

    samples = load_loose_jsonl(jsonl_path)
    class2idx_l1 = build_class_index(samples, "label_level1")
    class2idx_l2 = build_class_index(samples, "label_level2")
    if not class2idx_l1 or not class2idx_l2:
        raise ValueError("真实标签为空，无法计算 F1")
    

    y_true_l1, y_pred_l1, y_true_l2, y_pred_l2 = accumulate_arrays(
        samples, class2idx_l1, class2idx_l2
    )

    total_len = len(y_true_l1)
    indices = np.random.permutation(total_len)
    half_len = total_len // 2
    selected_indices = indices[:half_len]

    # 抽取一半数据
    y_true_l1 = y_true_l1[selected_indices]
    y_pred_l1 = y_pred_l1[selected_indices]
    y_true_l2 = y_true_l2[selected_indices]
    y_pred_l2 = y_pred_l2[selected_indices]

    metrics: Dict[str, float] = {}
    for avg in ("macro", "micro", "weighted"):
        metrics[f"f1_{avg}_l1"] = float(
            f1_score(y_true_l1, y_pred_l1, average=avg, zero_division=0)
        )
        metrics[f"f1_{avg}_l2"] = float(
            f1_score(y_true_l2, y_pred_l2, average=avg, zero_division=0)
        )

    # 若仍需返回 per-class F1，可在此添加
    return metrics



if __name__ == "__main__":
    import argparse, pprint
    parser = argparse.ArgumentParser(description="Compute two-level multi-label F1")
    parser.add_argument("-f", "--file", required=True, help="Path to generated_predictions.jsonl")
    args = parser.parse_args()

    res = compute_two_level_f1(args.file)
    pprint.pprint(res, width=120)
