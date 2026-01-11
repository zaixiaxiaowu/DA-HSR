#!/bin/bash
set -e

echo "=================================================="
echo "Starting Data Processing for all datasets"
echo "=================================================="

echo "Processing cloth (Clothing & Sports)..."
python data/handle.py \
    --dataset cloth \
    --domain_a "Clothing_Shoes_and_Jewelry" \
    --domain_b "Sports_and_Outdoors" \
    --start_time 1514736000 \
    --end_time 1577808000 \
    --user_core 5 \
    --item_core 3

echo "--------------------------------------------------"

echo "Processing Elec (Electronics & Cell Phones)..."
python data/handle.py \
    --dataset elec \
    --domain_a "Electronics" \
    --domain_b "Cell_Phones_and_Accessories" \
    --start_time 1500000000 \
    --end_time 1600000000 \
    --user_core 5 \
    --item_core 3

echo "--------------------------------------------------"

echo "Processing Food (Home & Grocery)..."
python data/handle.py \
    --dataset food \
    --domain_a "Home_and_Kitchen" \
    --domain_b "Grocery_and_Gourmet_Food" \
    --start_time 1500000000 \
    --end_time 1600000000 \
    --user_core 5 \
    --item_core 3

echo "=================================================="
echo "All datasets processed successfully!"
echo "Outputs are located in data/{dataset}/handled/"
echo "=================================================="