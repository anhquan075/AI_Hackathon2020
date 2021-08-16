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
        ```
        node crawler.js
        ```
    - Download images:
        ```
        python3 download.py
        ```
### II. Download data 
- Download the training set and test set via the following command below:      
  - Install ```gdown``` library:  
  ```
  pip3 install gdown
  ```
  - Download the "raw" training set (The training set is provided from the competition):
  ```
  gdown --id 1tlmVWM4mnfsELzLmNZgNhlBzy4CDVhRG
  ```
  - Download the "full" training set (Including the "raw" training set with crawling and batch-cropping data):
  ```
  gdown --id 1R4aN4UwTzHpbA3qHCmtWnt4YPve1rJpn
  ```
  - Download the test set: 
  ```
  gdown --id 1p3VK1LTcRtCKPkxp1McfVCSPnmfDiK8d
  ```

---
# Train model
## Training
1. Train on Colab
    - Link colab: [link](https://github.com/anhquan075/AI_Hackathon2020/blob/master/src/train_model/train_model.ipynb).  
2. Train with tensorflow
    - Link: [train_efficientNet_mutil_gpu.py](https://github.com/anhquan075/AI_Hackathon2020/blob/master/src/train_model/train_efficientNet_mutil_gpu.py).
3. Train with pytorch
    - Repo: [pytorch-image-models](https://github.com/rwightman/pytorch-image-models).
    - Train:
    
            ./distributed_train.sh <number_gpu> <path/to/dataset> --model <model_name></model_name> --pretrained --num-classes <number_class> -b <batch_size> --sched step --epochs <number_epoch> --decay-epochs 1 --decay-rate .9 --opt <optimizer> --opt-eps .001 -j 8 --warmup-lr <start_learning_rate> --weight-decay 1e-5 --lr <max_learning_rate>
    - Infer:

            CUDA_VISIBLE_DEVICES=<number> python3 inference.py <path/to/dataset> -b <batch_size> --num-classes <number_class> --model <model_name> --checkpoint <path/to/model>
    - **Train with SAM optimizer:** SAM simultaneously minimizes loss value and loss sharpness. In particular, it seeks parameters that lie in neighborhoods having uniformly low loss. SAM improves model generalization and yields [SoTA performance for several datasets](https://arxiv.org/pdf/2010.01412v1.pdf). Additionally, it provides robustness to label noise on par with that provided by SoTA procedures that specifically target learning with noisy labels ([Link repo](https://github.com/davda54/sam)).
## Inference 
Run ```python3 src/full_flow.py``` to predict with full flow from pre-processing to post-processing. Besides, you can use the service mentioned in Section [Service](#run-service) with UI interaction.

---
# Visualization 
- All code to visualize the prediction are located in ```vismatrix``` folder. Run ```python3 vismatrix/web.py``` to use.
- ### Usage:
  - The service asks two paths are test set path and ```visual.txt``` file path. For example, the ```visual.txt``` is located in ```/home/Downloads/visual.txt```, you just need fill in the ```labelpath``` form is ```/home/Downloads```, the ```imgpath``` is the path located your test set like ```/home/Downloads/test_set_A_full```.
  - The ```visual.txt``` file has the following format for the test set in [Section II](#ii-download-data), you can download the template to use in [here](https://drive.google.com/file/d/16HhcDE-hDmHB8jJIsHHCNyxJ9wJt1dRC/view?usp=sharing):
  
  ```
        Img00001.JPEG	0	1.0	0	0.93783253
        Img00002.JPEG	0	0.6196056	6	0.10300595
        Img00006.JPEG	0	0.97495306	7	0.0078060715
        Img00007.jpg	0	1.0	5	0.8099797
        Img00008.jpg	5	0.39101785	3	0.2744701
        Img00010.jpg	7	0.6946699	4	0.10549918
        Img00011.jpg	1	0.013945208	0	0.9652487
        Img00012.JPEG	0	1.0	0	0.98315215
        Img00013.JPEG	0	1.0	7	0.44441825
        Img00014.JPEG	0	1.0	0	0.76019937
        Img00015.JPEG	0	0.9101331	7	0.029524248
        Img00016.JPEG	0	1.0	0	0.78398454
        Img00017.jpg	0	0.9966485	7	0.0010322712
        Img00018.jpg	0	0.8327981	3	0.04957785
        Img00019.jpg	5	0.99875116	6	0.0006665691
        Img00020.JPEG	0	0.27356783	6	0.26584625
        Img00021.jpg	0	0.66508734	6	0.063248254
        Img00022.JPEG	0	0.97181964	7	0.
        ....
        ....
  ```


# Run service
This is the service that we used in the final round of the competition. 
```
python3 service.py
```
---

## Build and run docker
We provide a ```Dockerfile``` to easily install and run the service. You need two GPUs with at least 16GB VRAM to run this service. 
1. Build docker
    - **Create dockerfile**: [link Dockerfile](https://github.com/anhquan075/AI_Hackathon2020/blob/master/Dockerfile).
    - **Build image:**

            docker build -t <name>:<version> .

2. Run Docker

    - **Run image:**

            docker run --name <name_image> --gpus all -p <port>:8080 -v <path/to/local_dir>:<path/to/image_dir> -it <name>:<version> 

    - You can modify shm size by passing the optional parameter *--shm-size* to train the model with PyTorch. The default is 64MB.

            docker run --name <name_image> --gpus all -p <port>:8080 -v <path/to/local_dir>:<path/to/image_dir> -it <name>:<version> --shm-size=2048m
