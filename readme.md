# CDSR Project

本项目提供了 CDSR 模型的运行指南。请按照以下步骤配置环境并运行实验。

首先，请确保终端处于项目根目录下：
```bash
cd CDSR

一. 快速开始 (已有 SID)
如果你已经生成好了 SID (State ID)，可以直接使用以下步骤运行实验。

1. 环境配置
conda env create -f cdsr_env.yaml
conda activate cdsr

2. 运行实验
bash experiments/train_model_cloth.sh


二. 从中间开始（自行生成SID）
如果需要重新训练并自行生成 SID，请按照以下流程操作。

1. 环境配置
conda env create -f sid_env.yaml
conda activate sid

2. 训练 SID
bash experiments/train_sid_cloth.sh

3. 生成 SID
bash experiments/gen_sid_cloth.sh

4. 移动文件
生成SID后，找到生成的 reconstructed_emb.pkl 文件，将其移动到对应的处理目录：data/cloth/handled/

5. 跳转到一、快速开始


三. 从头开始（自行生成emb）

1. 环境配置
conda env create -f cdsr_env.yaml
conda activate cdsr

2. 数据集下载
将原始数据集放至data/{cloth/elec/food}/raw/下面

(1)cloth
wget -c https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFiles/Clothing_Shoes_and_Jewelry.json.gz
wget -c https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/metaFiles2/meta_Clothing_Shoes_and_Jewelry.json.gz
wget -c https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFiles/Sports_and_Outdoors.json.gz
wget -c https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/metaFiles2/meta_Sports_and_Outdoors.json.gz

(2)elec
wget -c https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFiles/Electronics.json.gz
wget -c https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/metaFiles2/meta_Electronics.json.gz
wget -c https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFiles/Cell_Phones_and_Accessories.json.gz
wget -c https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/metaFiles2/meta_Cell_Phones_and_Accessories.json.gz

(3)food：
wget -c https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFiles/Grocery_and_Gourmet_Food.json.gz
wget -c https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/metaFiles2/meta_Grocery_and_Gourmet_Food.json.gz
wget -c https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFiles/Home_and_Kitchen.json.gz
wget -c https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/metaFiles2/meta_Home_and_Kitchen.json.gz

3. 数据处理
bash experiments/handle.sh

4. 生成emb
见注意
配置 API KEY
bash experiments/gen_emb.sh

5. 跳转到二、从中间开始

四、其他数据集
将上面的cloth替换为elec\food

1. elec数据集
bash experiments/train_sid_elec.sh
bash experiments/gen_sid_elec.sh
bash experiments/train_model_elec.sh

2. food数据集
bash experiments/train_sid_fodd.sh
bash experiments/gen_sid_food.sh
bash experiments/train_model_food.sh


五、注意

Sports数据中缺失了一条元数据 ”B001ARYU58“

需再生成 data/cloth/handled/item2attributes_B.json 后将下面内容手动补充到该文件中，然后再执行generate_emb.sh：

"B001ARYU58": {
  "category": [
    "Sports & Outdoors",
    "Exercise & Fitness",
    "Strength Training Equipment",
    "Weights & Accessories",
    "Dumbbells"
  ],
  "tech1": "",
  "description": [
    "Adjustable Resistance: Each dumbbell adjusts from 5 to 52.5 pounds; adjusts in 2.5-pound increments up to the first 25 pounds.",
    "Efficient to Use: Lets you rapidly switch from one exercise to the next; combines 15 sets of weights into one, using a unique dial system.",
    "Space Efficient: Eliminates the need for multiple dumbbells cluttering your workout space.",
    "Quieter Workouts: Durable molding around metal plates provides for a smooth lift-off and quieter workouts... no clanging metal plates.",
    "Total Body Results: With the wider weight range you can do heavy exercises such as shrugs and lunges as well as lighter exercises such as curls and raises.",
    "Premium Grips: Flexible and durable composition materials."
  ],
  "fit": "",
  "title": "Bowflex SelectTech 552 Adjustable Dumbbells (Pair)",
  "also_buy": [
    "B000BH283K",
    "B000NZO0IE",
    "B07621LGTT",
    "B0751CLHQ6",
    "B071ZLBSJB"
  ],
  "tech2": "",
  "brand": "Bowflex",
  "feature": [
    "Adjustable from 5 to 52.5 lbs per dumbbell",
    "Replaces 15 sets of weights",
    "Weight selection dial for easy adjustment",
    "Space-saving design",
    "Durable molding around metal plates for quiet workouts",
    "Includes storage trays",
    "Works with JRNY app for motion tracking and trainer-led workouts"
  ],
  "rank": "Top 10 in Strength Training Dumbbells",
  "also_view": [
    "B001ARYU58",
    "B000NZO0IE",
    "B08KY6J4P4",
    "B09715D3W3"
  ],
  "details": {
    "Product Dimensions:": "16.9 x 8.3 x 9 inches ; 52.5 pounds",
    "Shipping Weight:": "115 pounds (approx for pair)",
    "ASIN:": "B001ARYU58",
    "Item model number:": "100182",
    "Material:": "Steel, Nylon, ABS, Thermoplastic Rubber"
  },
  "main_cat": "Sports & Outdoors",
  "similar_item": "",
  "date": "",
  "price": "$549.00",
  "asin": "B001ARYU58",
  "imageURL": [
    "https://m.media-amazon.com/images/I/71+pOdQ7iKL._AC_SL1500_.jpg",
    "https://m.media-amazon.com/images/I/71f-rGg-KNL._AC_SL1500_.jpg",
    "https://m.media-amazon.com/images/I/81ut0B-tVCL._AC_SL1500_.jpg",
    "https://m.media-amazon.com/images/I/81x%2B9%2B2%2B1%2BL._AC_SL1500_.jpg"
  ],
  "imageURLHighRes": [
    "https://m.media-amazon.com/images/I/71+pOdQ7iKL.jpg",
    "https://m.media-amazon.com/images/I/71f-rGg-KNL.jpg",
    "https://m.media-amazon.com/images/I/81ut0B-tVCL.jpg",
    "https://m.media-amazon.com/images/I/81x%2B9%2B2%2B1%2BL.jpg"
  ]
},