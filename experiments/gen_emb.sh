#!/bin/bash

API_KEY=""

python data/generate_emb.py --dataset cloth --api_key "$API_KEY"
python data/generate_emb.py --dataset elec --api_key "$API_KEY"
python data/generate_emb.py --dataset food --api_key "$API_KEY"