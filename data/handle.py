import argparse
import ast
import collections
import gzip
import json
import os
import pickle

from tqdm import tqdm


def parse_line(line):
    line = line.strip()
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return ast.literal_eval(line)


def open_text(path):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def load_reviews(path, start_time, end_time):
    user_items = collections.defaultdict(list)
    with open_text(path) as handle:
        for line in tqdm(handle, desc="Load reviews {}".format(os.path.basename(path))):
            row = parse_line(line)
            timestamp = int(row.get("unixReviewTime", 0))
            if timestamp < start_time or timestamp > end_time:
                continue
            user_items[row["reviewerID"]].append((row["asin"], timestamp))
    return user_items


def load_meta(path):
    meta = {}
    if not os.path.exists(path):
        return meta
    with open_text(path) as handle:
        for line in tqdm(handle, desc="Load meta {}".format(os.path.basename(path))):
            row = parse_line(line)
            asin = row.get("asin")
            if asin:
                meta[asin] = row
    return meta


def iterative_core(user_items, user_core=5, item_core=5):
    while True:
        user_items = {user: seq for user, seq in user_items.items() if len(seq) >= user_core}
        item_count = collections.Counter(item for seq in user_items.values() for item, _ in seq)
        keep_items = {item for item, count in item_count.items() if count >= item_core}
        filtered = {}
        for user, seq in user_items.items():
            kept = [(item, time) for item, time in seq if item in keep_items]
            if len(kept) >= user_core:
                filtered[user] = kept
        if len(filtered) == len(user_items) and all(len(filtered[u]) == len(user_items[u]) for u in filtered):
            return filtered
        user_items = filtered


def sort_unique(seq):
    seq = sorted(seq, key=lambda x: x[1])
    seen = set()
    ordered = []
    for item, time in seq:
        if item in seen:
            continue
        seen.add(item)
        ordered.append((item, time))
    return ordered


def remap_items(user_items):
    items = sorted({item for seq in user_items.values() for item, _ in seq})
    str2id = {asin: idx + 1 for idx, asin in enumerate(items)}
    id2str = {idx + 1: asin for idx, asin in enumerate(items)}
    remapped = {}
    for user, seq in user_items.items():
        remapped[user] = [(str2id[item], time) for item, time in sort_unique(seq)]
    return remapped, str2id, id2str


def build_attributes(asins, meta):
    attributes = {}
    for asin in asins:
        row = meta.get(asin, {})
        description = row.get("description", [])
        if isinstance(description, str):
            description = [description]
        attributes[asin] = {
            "title": row.get("title", asin),
            "brand": row.get("brand", ""),
            "category": row.get("category", row.get("categories", [])),
            "description": description,
            "asin": asin,
        }
    return attributes


def parse_args():
    parser = argparse.ArgumentParser(description="Amazon CDSR preprocessing")
    parser.add_argument("--dataset", type=str, required=True, choices=["cloth", "elec", "food"])
    parser.add_argument("--domain_a", type=str, required=True)
    parser.add_argument("--domain_b", type=str, required=True)
    parser.add_argument("--start_time", type=int, required=True)
    parser.add_argument("--end_time", type=int, required=True)
    parser.add_argument("--user_core", type=int, default=5)
    parser.add_argument("--item_core", type=int, default=5)
    parser.add_argument("--data_root", type=str, default="data")
    return parser.parse_args()


def main():
    args = parse_args()
    raw_dir = os.path.join(args.data_root, args.dataset, "raw")
    handled_dir = os.path.join(args.data_root, args.dataset, "handled")
    os.makedirs(handled_dir, exist_ok=True)

    reviews_a = os.path.join(raw_dir, "{}.json.gz".format(args.domain_a))
    reviews_b = os.path.join(raw_dir, "{}.json.gz".format(args.domain_b))
    meta_a = os.path.join(raw_dir, "meta_{}.json.gz".format(args.domain_a))
    meta_b = os.path.join(raw_dir, "meta_{}.json.gz".format(args.domain_b))

    data_a = iterative_core(load_reviews(reviews_a, args.start_time, args.end_time), args.user_core, args.item_core)
    data_b = iterative_core(load_reviews(reviews_b, args.start_time, args.end_time), args.user_core, args.item_core)
    data_a, item_str2id_a, item_id2str_a = remap_items(data_a)
    data_b, item_str2id_b, item_id2str_b = remap_items(data_b)

    users = sorted(set(data_a) | set(data_b))
    user_str2id = {user: idx + 1 for idx, user in enumerate(users)}
    user_id2str = {idx + 1: user for idx, user in enumerate(users)}

    inter_seq, domain_seq = {}, {}
    for user in users:
        seq_a = data_a.get(user, [])
        seq_b = data_b.get(user, [])
        merged = [(item, time, 0) for item, time in seq_a] + [(item, time, 1) for item, time in seq_b]
        merged = sorted(merged, key=lambda x: x[1])
        uid = user_str2id[user]
        inter_seq[uid] = [item for item, _, _ in merged]
        domain_seq[uid] = [domain for _, _, domain in merged]

    id_map = {
        "user_dict": {"str2id": user_str2id, "id2str": {str(k): v for k, v in user_id2str.items()}},
        "item_dict": {
            "str2id": {"0": item_str2id_a, "1": item_str2id_b},
            "id2str": {
                "0": {str(k): v for k, v in item_id2str_a.items()},
                "1": {str(k): v for k, v in item_id2str_b.items()},
            },
            "item_count": {"0": len(item_str2id_a), "1": len(item_str2id_b)},
        },
    }

    pair_name = {
        "cloth": "cloth_sport",
        "elec": "elec_phone",
        "food": "food_kitchen",
    }[args.dataset]
    pickle.dump((inter_seq, domain_seq), open(os.path.join(handled_dir, "{}.pkl".format(pair_name)), "wb"))
    json.dump(id_map, open(os.path.join(handled_dir, "id_map.json"), "w", encoding="utf-8"))

    attributes_a = build_attributes(item_str2id_a.keys(), load_meta(meta_a))
    attributes_b = build_attributes(item_str2id_b.keys(), load_meta(meta_b))
    json.dump(attributes_a, open(os.path.join(handled_dir, "item2attributes_A.json"), "w", encoding="utf-8"))
    json.dump(attributes_b, open(os.path.join(handled_dir, "item2attributes_B.json"), "w", encoding="utf-8"))

    print("Users:", len(users))
    print("Items A/B:", len(item_str2id_a), len(item_str2id_b))
    print("Saved to", handled_dir)


if __name__ == "__main__":
    main()
