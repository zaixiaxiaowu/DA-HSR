import argparse
import json
import os
import pickle
import time

import numpy as np
from tqdm import tqdm


REASONING_PROMPT = """You are an expert E-commerce Data Analyst. Your task is to reason about the provided item attributes and generate a semantically rich summary. Analyze the item based on the following dimensions: (1) Core Characteristics: Extract the most critical specifications and selling points. (2) Usage Scenarios: Infer who would use, wear, or consume the item and in what context. (3) Implied Value: Identify hidden advantages based on the brand or materials.

Item attributes:
{item_text}
"""


def item_to_text(attr):
    title = attr.get("title", "")
    brand = attr.get("brand", "")
    category = attr.get("category", [])
    if isinstance(category, list):
        category = " > ".join([str(x) for x in category])
    description = attr.get("description", [])
    if isinstance(description, list):
        description = " ".join([str(x) for x in description])
    return "Title: {}\nBrand: {}\nCategory: {}\nDescription: {}".format(title, brand, category, description)


def ordered_asins(id_map, domain):
    id2str = id_map["item_dict"]["id2str"][domain]
    count = id_map["item_dict"]["item_count"][domain]
    asins = []
    for idx in range(1, count + 1):
        asins.append(id2str[str(idx)])
    return asins


def call_llm(api_base, api_key, model_name, prompt, max_retry=3):
    import requests

    url = api_base.rstrip("/") + "/chat/completions"
    headers = {"Authorization": "Bearer {}".format(api_key), "Content-Type": "application/json"}
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
    }
    for attempt in range(max_retry):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            if attempt == max_retry - 1:
                print("LLM call failed:", exc)
                return ""
            time.sleep(2 ** attempt)
    return ""


def embed_texts(texts, model_name):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
    return np.asarray(embeddings, dtype=np.float32)


def parse_args():
    parser = argparse.ArgumentParser(description="Reasoning-enhanced static semantic encoding")
    parser.add_argument("--dataset", type=str, required=True, choices=["cloth", "elec", "food"])
    parser.add_argument("--api_key", type=str, default="")
    parser.add_argument("--llm_api_base", type=str, default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--llm_model", type=str, default="qwen3-max")
    parser.add_argument("--embed_model", type=str, default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--data_root", type=str, default="data")
    return parser.parse_args()


def main():
    args = parse_args()
    handled_dir = os.path.join(args.data_root, args.dataset, "handled")
    id_map = json.load(open(os.path.join(handled_dir, "id_map.json"), "r", encoding="utf-8"))
    attr_a = json.load(open(os.path.join(handled_dir, "item2attributes_A.json"), "r", encoding="utf-8"))
    attr_b = json.load(open(os.path.join(handled_dir, "item2attributes_B.json"), "r", encoding="utf-8"))

    records = []
    for domain, attributes in [("0", attr_a), ("1", attr_b)]:
        for asin in ordered_asins(id_map, domain):
            records.append((asin, attributes.get(asin, {"title": asin, "asin": asin})))

    traces_path = os.path.join(handled_dir, "reasoning_traces.json")
    traces = {}
    if os.path.exists(traces_path):
        traces = json.load(open(traces_path, "r", encoding="utf-8"))

    if args.api_key:
        for asin, attr in tqdm(records, desc="Reasoning traces"):
            if asin in traces and traces[asin]:
                continue
            prompt = REASONING_PROMPT.format(item_text=item_to_text(attr))
            traces[asin] = call_llm(args.llm_api_base, args.api_key, args.llm_model, prompt)
            if len(traces) % 50 == 0:
                json.dump(traces, open(traces_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        json.dump(traces, open(traces_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    else:
        print("No API key provided; encoding raw item text without reasoning traces.")

    encoded_inputs = []
    for asin, attr in records:
        raw = item_to_text(attr)
        trace = traces.get(asin, "")
        encoded_inputs.append(raw if not trace else "{}\nReasoning: {}".format(raw, trace))

    embeddings = embed_texts(encoded_inputs, args.embed_model)
    out_all = os.path.join(handled_dir, "itm_emb_np_all.pkl")
    pickle.dump(embeddings, open(out_all, "wb"))
    print("Saved {} embeddings of dim {} to {}".format(embeddings.shape[0], embeddings.shape[1], out_all))


if __name__ == "__main__":
    main()
