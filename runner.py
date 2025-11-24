#!/usr/bin/env python3
"""
run_vsi_videollama_eval.py
End-to-end orchestrator for VSI-Bench prompt arms using VideoLLaMA3-7B.
Edit CONFIG below before running.
"""

import os
import json
import time
import random
from typing import List, Dict, Any, Callable
from datasets import load_dataset

# ---- CONFIG ----
HF_DATASET_ID = "vsi-bench/vsi-bench"        # Hugging Face dataset id for VSI-Bench
DATA_SPLIT = "test"                          # dataset split to use
OUTDIR = "vsi_videollama_results"
MAX_SAMPLES = 10                             # set to int for quick pilot (e.g., 100) - CHANGE TO None FOR FULL RUN
SEED = 42
SPATIAL_TASK_TYPES = {"relative_distance", "absolute_distance", "route", "configurational"}  # adjust to dataset schema

# VideoLLaMA model config
MODEL_ID = "DAMO-NLP-SG/VideoLLaMA3-7B"
USE_GPU = True                               # set False to run on CPU (very slow)
GEN_MAX_NEW_TOKENS = 128
GEN_DO_SAMPLE = False
GEN_TEMPERATURE = 0.0

# Tune processor video input
DEFAULT_FPS = 1
DEFAULT_MAX_FRAMES = 8

try:
    from eval_helpers import compute_metrics, load_gold  # change to actual module from repo
except Exception:
    compute_metrics = None
    load_gold = None
    print("Warning: eval_helpers not importable. Script will fallback to simple ACC for MC tasks.")

# Model imports deferred to model loader function to avoid heavy import when not needed
def load_videollama_model(model_id: str):
    """
    Load VideoLLaMA3-7B with trust_remote_code and device_map auto.
    Returns (model, processor). Caller should handle mixed precision & device placement.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    # The model card recommends bfloat16 + trust_remote_code; modify if you don't have bfloat support
    kwargs = {
        "trust_remote_code": True,
        "device_map": "auto"
    }
    # prefer bfloat16 if available on device; fallback to float16 if not
    if torch.cuda.is_available():
        kwargs["torch_dtype"] = torch.bfloat16 if hasattr(torch, "bfloat16") else torch.float16

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model.eval()
    return model, processor

# Prompt templates
PROMPT_TEMPLATES = {
    "baseline": "{question}\n\nAnswer:",
    "cot": "Think step by step. {question}\n\nAnswer:",
    "topdown_map": ("First write a compact top-down layout labeled 'Map:' using a 10x10 grid or a list of 'object: (x,y)'. "
                    "Keep Map to at most 6 lines. Then give the final answer after 'Answer:'\n\n{question}\n\nMap:\n\nAnswer:"),
    "relative_anchor": ("First list distances between anchor object pairs under 'Relatives:' (max 6 lines). "
                        "Then write the final answer after 'Answer:'\n\n{question}\n\nRelatives:\n\nAnswer:"),
    "camera_traj": ("First list camera movement steps under 'Trajectory:' (max 6 lines). "
                    "Then give the final answer after 'Answer:'\n\n{question}\n\nTrajectory:\n\nAnswer:"),
    "hybrid": ("First provide a short 'Map:' (max 6 lines), then 'Relatives:' (pairwise distances, max 4 lines). "
               "Finally give the answer after 'Answer:'\n\n{question}\n\nMap:\n\nRelatives:\n\nAnswer:")
}

def ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)

def validate_dataset_sample(sample: Dict[str, Any], idx: int) -> bool:
    """
    Validate that a dataset sample has the required fields.
    Returns True if valid, False otherwise.
    """
    required_fields = ["question", "prompt", "text"]
    has_question = any(sample.get(field) for field in required_fields)
    
    if not has_question:
        print(f"Warning: Sample {idx} missing question field")
        return False
    
    video_fields = ["video_path", "video", "video_path_local", "media"]
    has_video = any(sample.get(field) for field in video_fields)
    
    if not has_video:
        print(f"Warning: Sample {idx} missing video field")
        return False
    
    return True

def sample_spatial_subset(ds, max_samples=None, seed=SEED):
    random.seed(seed)
    rows = []
    for idx, item in enumerate(ds):
        # adapt these keys to actual dataset schema
        task_type = item.get("task_type") or item.get("type") or ""
        if task_type in SPATIAL_TASK_TYPES:
            # Validate sample before adding
            if validate_dataset_sample(item, idx):
                rows.append(item)
            else:
                print(f"Skipping invalid sample at index {idx}")
    
    if max_samples:
        rows = random.sample(rows, min(len(rows), max_samples))
    return rows

def render_prompt(template: str, question_text: str) -> str:
    return template.format(question=question_text)

def parse_answer(model_text: str) -> str:
    """
    Extract text after 'Answer:' (case-insensitive). Fallback to last non-empty line.
    """
    if not model_text:
        return ""
    lines = [l.strip() for l in model_text.splitlines() if l.strip()]
    for ln in lines:
        if ln.lower().startswith("answer:"):
            return ln.split(":", 1)[1].strip()
    return lines[-1].strip()

def run_arm(samples: List[Dict[str, Any]], arm_name: str, template: str,
            model_fn: Callable[[Dict[str, Any]], str], outdir: str):
    results = []
    t0 = time.time()
    for i, s in enumerate(samples):
        q_id = s.get("id") or s.get("example_id") or f"idx_{i}"
        # adapt dataset fields below to actual HF dataset schema
        question_text = s.get("question") or s.get("prompt") or s.get("text") or ""
        gold = s.get("answer") or s.get("labels") or s.get("gold") or None
        prompt = render_prompt(template, question_text)

        # prepare model input package for model_fn (video path + prompt + sampling params)
        video_path = s.get("video_path") or s.get("video") or s.get("video_path_local") or s.get("media", {}).get("video_path") if isinstance(s.get("media"), dict) else None

        # Validate video path exists
        if not video_path:
            print(f"Warning: No video path found for sample {q_id}, skipping...")
            results.append({
                "q_id": q_id,
                "task_type": s.get("task_type", "unknown"),
                "question": question_text,
                "gold": gold,
                "prompt": prompt,
                "video_path": None,
                "raw": "ERROR: No video path provided",
                "parsed": "",
                "error": "missing_video_path"
            })
            continue

        model_input = {
            "prompt": prompt,
            "video_path": video_path,
            "fps": DEFAULT_FPS,
            "max_frames": DEFAULT_MAX_FRAMES,
            "gen_max_new_tokens": GEN_MAX_NEW_TOKENS,
            "do_sample": GEN_DO_SAMPLE,
            "temperature": GEN_TEMPERATURE
        }

        # Wrap model inference in try-except to handle errors gracefully
        try:
            raw = model_fn(model_input)
            parsed = parse_answer(raw)
            error = None
        except Exception as e:
            print(f"Error processing sample {q_id}: {str(e)}")
            raw = f"ERROR: {str(e)}"
            parsed = ""
            error = str(e)

        results.append({
            "q_id": q_id,
            "task_type": s.get("task_type", "unknown"),
            "question": question_text,
            "gold": gold,
            "prompt": prompt,
            "video_path": video_path,
            "raw": raw,
            "parsed": parsed,
            **({"error": error} if error else {})
        })

        if (i + 1) % 20 == 0 or (i + 1) == len(samples):
            elapsed = time.time() - t0
            print(f"[{arm_name}] {i+1}/{len(samples)} elapsed {elapsed:.1f}s")

    outpath = os.path.join(outdir, f"results_{arm_name}.jsonl")
    with open(outpath, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved {len(results)} results for arm {arm_name} to {outpath}")
    return results

def compute_and_print_metrics(results: List[Dict[str, Any]], arm_name: str):
    if compute_metrics:
        try:
            preds = [r["parsed"] for r in results]
            golds = [r["gold"] for r in results]
            metrics = compute_metrics(preds, golds)
            print(f"{arm_name} metrics: {metrics}")
            return metrics
        except Exception as e:
            print("compute_metrics call failed:", e)

    # fallback: simple exact-match ACC for MC tasks
    total = 0
    correct = 0
    for r in results:
        if r["gold"] is None:
            continue
        total += 1
        if str(r["parsed"]).strip().lower() == str(r["gold"]).strip().lower():
            correct += 1
    acc = correct / total if total else 0.0
    print(f"[fallback] {arm_name} ACC = {acc:.4f} ({correct}/{total})")
    return {"ACC": acc, "correct": correct, "total": total}

def main():
    ensure_outdir(OUTDIR)
    random.seed(SEED)

    print("Loading dataset:", HF_DATASET_ID, "split:", DATA_SPLIT)
    ds = load_dataset(HF_DATASET_ID, split=DATA_SPLIT)
    print("Dataset size:", len(ds))

    samples = sample_spatial_subset(ds, max_samples=MAX_SAMPLES)
    print("Spatial subset size:", len(samples))
    if len(samples) == 0:
        print("No spatial samples found. Check dataset schema and SPATIAL_TASK_TYPES.")
        return

    # Load model and processor
    print("Loading VideoLLaMA model:", MODEL_ID)
    model, processor = load_videollama_model(MODEL_ID)

    # Model inference function using loaded model & processor
    import torch
    def model_fn(model_input: Dict[str, Any]) -> str:
        """
        model_input keys: prompt, video_path, fps, max_frames, gen_max_new_tokens, do_sample, temperature
        Returns raw generated text from model (string).
        """
        prompt = model_input["prompt"]
        video_path = model_input.get("video_path")
        fps = model_input.get("fps", DEFAULT_FPS)
        max_frames = model_input.get("max_frames", DEFAULT_MAX_FRAMES)
        gen_max_new_tokens = model_input.get("gen_max_new_tokens", GEN_MAX_NEW_TOKENS)
        do_sample = model_input.get("do_sample", GEN_DO_SAMPLE)
        temperature = model_input.get("temperature", GEN_TEMPERATURE)

        # Validate video path
        if not video_path or not os.path.exists(video_path):
            raise ValueError(f"Invalid video path: {video_path}")

        # Build conversation according to VideoLLaMA processor expectations
        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [
                {"type": "video", "video": {"video_path": video_path, "fps": fps, "max_frames": max_frames}},
                {"type": "text", "text": prompt}
            ]}
        ]

        inputs = processor(conversation=conversation, return_tensors="pt")
        # move tensors to device(s)
        inputs = {k: (v.cuda() if isinstance(v, torch.Tensor) and USE_GPU else v) for k, v in inputs.items()}
        # optionally cast pixel values to model dtype if present
        if "pixel_values" in inputs:
            # match dtype to model
            dtype = next(model.parameters()).dtype
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype)

        with torch.no_grad():
            out_ids = model.generate(**inputs,
                                     max_new_tokens=gen_max_new_tokens,
                                     do_sample=do_sample,
                                     temperature=temperature)
        
        # decode with processor (model card uses batch_decode)
        try:
            text = processor.batch_decode(out_ids, skip_special_tokens=True)[0].strip()
        except Exception as e:
            # fallback: try using tokenizer directly or return error message
            try:
                if hasattr(model, 'tokenizer'):
                    text = model.tokenizer.batch_decode(out_ids, skip_special_tokens=True)[0].strip()
                else:
                    text = f"ERROR: Failed to decode output - {str(e)}"
            except:
                text = f"ERROR: Failed to decode output - {str(e)}"
        
        # Clear GPU cache periodically to prevent OOM
        if USE_GPU and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return text

    # Run arms
    arms = list(PROMPT_TEMPLATES.keys())
    summary = {}
    for arm in arms:
        print("=" * 60)
        print("Running arm:", arm)
        results = run_arm(samples, arm, PROMPT_TEMPLATES[arm], model_fn, OUTDIR)
        metrics = compute_and_print_metrics(results, arm)
        summary[arm] = metrics

    summary_path = os.path.join(OUTDIR, "summary_metrics.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("Saved summary to", summary_path)
    print("Done.")

if __name__ == "__main__":
    main()
