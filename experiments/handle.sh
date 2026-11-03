#!/bin/bash
set -e

echo "=================================================="
echo "Starting data processing for all datasets"
echo "=================================================="

echo "Processing cloth (Clothing & Sport)..."
python data/handle.py \
    --dataset cloth \
    --domain_a "Clothing_Shoes_and_Jewelry" \
    --domain_b "Sports_and_Outdoors" \
    --start_time 1514736000 \
    --end_time 1577808000 \
    --user_core 5 \
    --item_core 5

echo "Processing elec (Electronics & Phone)..."
python data/handle.py \
    --dataset elec \
    --domain_a "Electronics" \
    --domain_b "Cell_Phones_and_Accessories" \
    --start_time 1500000000 \
    --end_time 1600000000 \
    --user_core 5 \
    --item_core 5

echo "Processing food (Food & Kitchen)..."
python data/handle.py \
    --dataset food \
    --domain_a "Grocery_and_Gourmet_Food" \
    --domain_b "Home_and_Kitchen" \
    --start_time 1500000000 \
    --end_time 1600000000 \
    --user_core 5 \
    --item_core 5

echo "=================================================="
echo "All datasets processed successfully."
echo "=================================================="
