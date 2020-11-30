# AICLUB_Hackathon2020

UIT AI Club at AI Hackathon 2020 @ AI4VN

---

# Prepare dataset
### I. Crawl data
1. Installation
    - **Install npm:** npm makes it easy for JavaScript developers to share and reuse code, and makes it easy to update the code that you’re sharing, so you can build amazing things ([link install](https://www.npmjs.com/get-npm)).  
    - **images-scraper:** This a simple way to scrape Google images using Puppeteer. The headless browser will behave as a 'normal' user and scrolls to the bottom of the page until there are enough results ([link repo](https://github.com/pevers/images-scraper)).  
    - npm install images-scraper  
2. Crawl
    - Run images-scraper:

            node crawler.js
    - Download images:

            python3 download.py

---
# Train model
1. Colab
    - Link colab: [link](https://github.com/anminhhung/AI_Hackathon2020/blob/master/src/train_model/train_model.ipynb).  
2. Train with tensorflow
    - Link: [train_efficientNet_mutil_gpu.py](https://github.com/anminhhung/AI_Hackathon2020/blob/master/src/train_model/train_efficientNet_mutil_gpu.py).
3. Train with pytorch
    - Repo: [pytorch-image-models](https://github.com/rwightman/pytorch-image-models).
    - Train:
    
            ./distributed_train.sh <number_gpu> <path/to/dataset> --model <model_name></model_name> --pretrained --num-classes <number_class> -b <batch_size> --sched step --epochs <number_epoch> --decay-epochs 1 --decay-rate .9 --opt <optimizer> --opt-eps .001 -j 8 --warmup-lr <start_learning_rate> --weight-decay 1e-5 --lr <max_learning_rate>
    - Infer:

            CUDA_VISIBLE_DEVICES=<number> python3 inference.py <path/to/dataset> -b <batch_size> --num-classes <number_class> --model <model_name> --checkpoint <path/to/model>
    - **Train with SAM optimizer:** SAM simultaneously minimizes loss value and loss sharpness. In particular, it seeks parameters that lie in neighborhoods having uniformly low loss. SAM improves model generalization and yields [SoTA performance for several datasets](https://arxiv.org/pdf/2010.01412v1.pdf). Additionally, it provides robustness to label noise on par with that provided by SoTA procedures that specifically target learning with noisy labels ([Link repo](https://github.com/davda54/sam)).

---

# Run service

        python3 service.py

---

# Build and run docker
1. Build docker
    - **Create dockerfile**: [link Dockerfile](https://github.com/anminhhung/AI_Hackathon2020/blob/master/Dockerfile).
    - **Build image:**

            docker build -t <name>:<version> .

2. Run Docker

    - **Run image:**

            docker run --name <name_image> --gpus all -p <port>:8080 -v <path/to/local_dir>:<path/to/image_dir> -it <name>:<version> 

    - You can modify shm size by passing the optional parameter *--shm-size* to train the model with PyTorch. The default is 64MB.

            docker run --name <name_image> --gpus all -p <port>:8080 -v <path/to/local_dir>:<path/to/image_dir> -it <name>:<version> --shm-size=2048m