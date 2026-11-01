"""Train DN-AI Micro and export a float16 checkpoint for the website runtime.

    python -m ai.dnai.train --steps 700
"""

from __future__ import annotations

import argparse
import base64
import gc
import json
import time
from pathlib import Path

import numpy as np

from ai.dnai.data import build_examples, knowledge_articles
from ai.dnai.generate import generate
from ai.dnai.model import Arch, forward_logits, init_params, load_params, loss_from_logits, numpy_params, zero_grads
from ai.dnai.tokenizer import Tokenizer

ROOT = Path(__file__).resolve().parents[2]
CKPT_PATH = ROOT / "src" / "lib" / "dnai" / "checkpoint.json"
REPORT_PATH = ROOT / "src" / "lib" / "dnai" / "report.json"
KNOW_PATH = ROOT / "src" / "lib" / "dnai" / "knowledge.json"
NPZ_PATH = ROOT / "ai" / "checkpoints" / "dnai-micro.npz"


def encode_example(row: dict, tok: Tokenizer, max_seq: int) -> tuple[list[int], list[int]] | None:
    ids: list[int] = [tok.bos_id]
    if row["knowledge"]:
        ids.append(tok.know_id)
        ids.extend(tok.encode(row["knowledge"]))
    for turn in row["history"]:
        ids.append(tok.user_id if turn["role"] == "user" else tok.asst_id)
        ids.extend(tok.encode(turn["content"]))
    ids.append(tok.user_id)
    ids.extend(tok.encode(row["user"]))
    asst_at = len(ids)
    ids.append(tok.asst_id)
    ids.extend(tok.encode(row["assistant"]))
    ids.append(tok.eos_id)
    if len(ids) > max_seq:
        overflow = len(ids) - max_seq
        ids = ids[overflow:]
        asst_at -= overflow
    if asst_at < 0 or asst_at >= len(ids) - 1:
        return None
    labels_mask_start = asst_at  # predict tokens after the asst marker, including eos
    return ids, labels_mask_start


def batch_from(encoded: list[tuple[list[int], int]], idxs: np.ndarray, pad: int):
    seqs = [encoded[i][0] for i in idxs]
    starts = [encoded[i][1] for i in idxs]
    width = max(len(s) for s in seqs)
    b = len(seqs)
    ids = np.full((b, width), pad, dtype=np.int64)
    labels = np.full((b, width - 1), pad, dtype=np.int64)
    mask = np.zeros((b, width - 1), dtype=np.float32)
    for i, seq in enumerate(seqs):
        ids[i, : len(seq)] = seq
        labels[i, : len(seq) - 1] = seq[1:]
        # Loss on positions t where the target token index (t+1) is strictly after <asst>.
        start = starts[i]
        for t in range(len(seq) - 1):
            if t + 1 > start:
                mask[i, t] = 1.0
    return ids[:, :-1], labels, mask


def f16_b64(arr: np.ndarray) -> dict:
    raw = np.ascontiguousarray(arr.astype(np.float16)).tobytes()
    return {"s": list(arr.shape), "d": base64.b64encode(raw).decode("ascii")}


def adam_step(params, moments, step, lr, wd):
    b1, b2, eps = 0.9, 0.999, 1e-8
    # clip
    total = 0.0
    for t in params.values():
        if t.grad is None:
            continue
        total += float(np.sum(t.grad * t.grad))
    scale = 1.0
    if total > 1.0:
        scale = 1.0 / (total ** 0.5)
    for name, t in params.items():
        if t.grad is None:
            continue
        g = t.grad * np.float32(scale)
        m, v = moments[name]
        m *= b1
        m += np.float32(1 - b1) * g
        v *= b2
        v += np.float32(1 - b2) * (g * g)
        mhat = m / np.float32(1 - b1**step)
        vhat = v / np.float32(1 - b2**step)
        upd = lr * mhat / (np.sqrt(vhat) + np.float32(eps))
        decay_names = (".wq", ".wk", ".wv", ".wo", ".w1", ".w2", "embed")
        if name == "embed" or name.endswith(decay_names):
            upd = upd + np.float32(lr * wd) * t.data
        t.data -= upd


def train(steps: int, seed: int, batch: int, lr: float) -> None:
    rows = build_examples()
    texts = []
    for row in rows:
        texts.append(row["user"])
        texts.append(row["assistant"])
        texts.append(row["knowledge"])
        for turn in row["history"]:
            texts.append(turn["content"])
    tok = Tokenizer.build(texts)
    arch = Arch(vocab_size=tok.vocab_size, d_model=128, n_layer=4, n_head=4, d_ff=384, max_seq=136)
    rng = np.random.default_rng(seed)
    params = init_params(arch, rng)
    held_out = {
        "How should a paragraph be built?",
        "Name the planets in order.",
        "What is self-attention?",
        "Tương quan có phải là nguyên nhân không?",
    }
    val_idx = [i for i, row in enumerate(rows) if row["user"] in held_out]
    # encoded is parallel to rows only when every row survives encode_example.
    # Rebuild val membership from encoded prompts is unnecessary: encode keeps order
    # of surviving rows. Track survival explicitly.
    encoded = []
    kept_users = []
    for row in rows:
        item = encode_example(row, tok, arch.max_seq)
        if item:
            encoded.append(item)
            kept_users.append(row["user"])
    if len(encoded) < 8:
        raise SystemExit("not enough training rows")
    val_idx = [i for i, user in enumerate(kept_users) if user in held_out]
    train_idx = [i for i in range(len(encoded)) if i not in set(val_idx)]
    moments = {k: (np.zeros_like(v.data), np.zeros_like(v.data)) for k, v in params.items()}
    print(f"vocab={tok.vocab_size} train={len(train_idx)} val={len(val_idx)} steps={steps}")
    t0 = time.time()
    last = 0.0
    for step in range(1, steps + 1):
        # cosine after warmup
        warm = 40
        if step < warm:
            cur_lr = lr * step / warm
        else:
            progress = (step - warm) / max(1, steps - warm)
            cur_lr = 1e-4 + 0.5 * (lr - 1e-4) * (1 + np.cos(np.pi * progress))
        pick = rng.choice(train_idx, size=min(batch, len(train_idx)), replace=False)
        inputs, labels, mask = batch_from(encoded, pick, tok.pad_id)
        zero_grads(params)
        logits = forward_logits(inputs, params, arch)
        loss = loss_from_logits(logits, labels, mask)
        loss.backward()
        adam_step(params, moments, step, float(cur_lr), 0.01)
        last = float(loss.data)
        loss._prev.clear()
        loss._bw = None
        del logits, loss, inputs, labels, mask
        if step % 5 == 0:
            gc.collect()
        if step % 20 == 0 or step == 1:
            print(f"step {step:4d} loss {last:.4f} lr {cur_lr:.5f} elapsed {time.time()-t0:.1f}s")
    val_loss = eval_loss(encoded, val_idx, params, arch, tok.pad_id, batch)
    arrays = numpy_params(params)
    samples = sample_report(tok, arrays, arch)
    train_meta = {
        "steps": steps,
        "seed": seed,
        "train_loss": round(last, 4),
        "val_loss": round(val_loss, 4),
        "val_perplexity": round(float(np.exp(min(val_loss, 20))), 3),
        "train_rows": len(train_idx),
        "val_rows": len(val_idx),
        "tokens_seen_estimate": steps * batch * 48,
        "seconds": round(time.time() - t0, 1),
    }
    export(tok, arch, arrays, train_meta, samples)
    print("val_loss", val_loss)
    for s in samples:
        print("---", s["prompt"])
        print(s["greedy"])


def eval_loss(encoded, idxs, params, arch, pad, batch) -> float:
    if not idxs:
        return 0.0
    total = 0.0
    count = 0
    for start in range(0, len(idxs), batch):
        chunk = np.asarray(idxs[start : start + batch])
        inputs, labels, mask = batch_from(encoded, chunk, pad)
        zero_grads(params)
        loss = loss_from_logits(forward_logits(inputs, params, arch), labels, mask)
        # mean already, weight by mask sum roughly via equal batches
        total += float(loss.data) * max(1, int(mask.sum()))
        count += max(1, int(mask.sum()))
        # drop graph
        for p in params.values():
            p._prev.clear()
    return total / max(1, count)


def build_prompt(tok: Tokenizer, user: str, knowledge: str, max_seq: int) -> list[int]:
    ids = [tok.bos_id]
    if knowledge:
        ids += [tok.know_id] + tok.encode(knowledge)
    ids += [tok.user_id] + tok.encode(user) + [tok.asst_id]
    if len(ids) > max_seq:
        ids = ids[-max_seq:]
        if ids[-1] != tok.asst_id:
            ids[-1] = tok.asst_id
    return ids


def sample_report(tok: Tokenizer, arrays, arch: Arch) -> list[dict]:
    prompts = [
        ("Xin chào", ""),
        ("DN-AI là gì?", "DN-AI là nền tảng trí tuệ nhân tạo tự phát triển. Slogan là Think Smarter. Solve Anything. Mô hình ngôn ngữ DN-AI Micro là Transformer do dự án tự huấn luyện, không gọi OpenAI, Gemini, Claude hay API chatbot khác."),
        ("Bạn có nhìn được ảnh không?", "DN-AI Micro là mô hình ngôn ngữ chữ, chưa được huấn luyện thị giác. Ảnh tải lên không được phân tích nội dung. Hệ thống phải nói rõ chưa có vision, không được đoán vật trong ảnh."),
        ("What is self-attention?", "Transformer xử lý chuỗi bằng self-attention. Mỗi token được nhúng thành vector, cộng thông tin vị trí bằng RoPE, rồi đi qua nhiều lớp attention nhiều đầu và mạng feed-forward. Mô hình nhân quả chỉ nhìn token phía trước khi sinh chữ."),
        ("Name the planets in order.", "Hệ Mặt Trời có Mặt Trời ở giữa và tám hành tinh: Thủy, Kim, Trái Đất, Hỏa, Mộc, Thổ, Thiên Vương, Hải Vương. Trái Đất là hành tinh thứ ba, có một vệ tinh tự nhiên là Mặt Trăng."),
        ("một câu chưa có trong tập huấn luyện về công thức nấu phở", ""),
    ]
    out = []
    for user, know in prompts:
        prompt = build_prompt(tok, user, know, arch.max_seq)
        greedy, glp = generate(prompt, arrays, arch, tok.eos_id, 60, 1e-4, 1, 1.0, 1)
        sampled, slp = generate(prompt, arrays, arch, tok.eos_id, 60, 0.8, 40, 0.9, 3)
        out.append(
            {
                "prompt": user,
                "greedy": tok.decode(greedy),
                "greedy_logprob": round(glp, 3),
                "sample": tok.decode(sampled),
                "sample_logprob": round(slp, 3),
            }
        )
    return out


def export(tok: Tokenizer, arch: Arch, arrays, train_meta: dict, samples: list[dict]) -> None:
    CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    NPZ_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(NPZ_PATH, **arrays)
    tensors = {name: f16_b64(arr) for name, arr in arrays.items()}
    payload = {
        "name": "dnai-micro",
        "version": 1,
        "arch": arch.to_dict(),
        "vocab": tok.id_to_token,
        "train": train_meta,
        "tensors": tensors,
        "license": "Project-original weights. Runtime math uses NumPy (BSD-3-Clause) for training.",
    }
    CKPT_PATH.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    KNOW_PATH.write_text(json.dumps(knowledge_articles(), ensure_ascii=False), encoding="utf-8")
    report = {
        "model": "dnai-micro",
        "status": "trained",
        "note": "DN-AI Micro is a small Transformer fit to the project's original seed set. Train loss can be very low while held-out wordings stay weak. It is not a commercial LLM and it does not call third-party chatbot APIs.",
        "train": train_meta,
        "arch": arch.to_dict(),
        "vocab_size": tok.vocab_size,
        "samples": samples,
        "param_count": int(sum(a.size for a in arrays.values())),
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", CKPT_PATH, "params", report["param_count"])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=700)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=2.5e-3)
    args = p.parse_args()
    train(args.steps, args.seed, args.batch, args.lr)


if __name__ == "__main__":
    main()
