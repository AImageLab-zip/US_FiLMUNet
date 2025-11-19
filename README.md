# US-FiLMUNet: Cross-Domain Ultrasound Segmentation

This repository contains the official PyTorch implementation for the paper: "[TOO BIG TO FAIL? NOT QUITE: FILM-UNET BEATS FOUNDATION MODELS IN CROSS-DOMAIN ULTRASOUND SEGMENTATION](https://arxiv.org/abs/YOUR_PAPER_ID)".

Here you will find the code to run our model in inference and to fine-tune it on your own ultrasound image dataset.

## Dataset

This work introduces an extention of the original [TesticulUS](https://ditto.ing.unimore.it/testiculus/) dataset, indeed among the original ~9,300 diffusion generated images **810** of them have been choosen by expert annotators as the most realistic and therefore have been annotated with segmentation masks.

### Dataset Card ~ TesticulUS-Syn
    - 810 sythetic images selected from the original **TesticulUS**
    - Segmentation Maks available for each image
    - In the presented work they proove themselfs valuable substitute to real images
    
You can find more information about the dataset at [https://ditto.ing.unimore.it/testiculus/](https://ditto.ing.unimore.it/testiculus/).

## Inference with our Pre-trained Model

Our pre-trained US-FiLMUNet model is available on Hugging Face at [AImageLab-Zip/US_FiLMUNet](https://huggingface.co/AImageLab-Zip/US_FiLMUNet).

You can easily load and use the model for inference on your own images.

First, make sure you have the required libraries installed:

```bash
pip install torch torchvision transformers
```

Then, you can use the following Python code to load the model and perform segmentation on a sample image:

```python
import torch
from PIL import Image
from transformers import AutoModel, AutoImageProcessor

# Load the model and processor from Hugging Face Hub
model = AutoModel.from_pretrained("AImageLab-Zip/US_FiLMUNet", trust_remote_code=True)
# Also the four stages version is availbale
# model_4_stages = AutoModel.from_pretrained(
#     "AImageLab-Zip/US_FiLMUNet", 
#     subfolder="unet_4_stages",
#     trust_remote_code=True
# )
processor = AutoImageProcessor.from_pretrained("AImageLab-Zip/US_FiLMUNet", trust_remote_code=True)
model.eval()

# Load your image
image_path = 'path/to/your/ultrasound_image.png'
image = Image.open(image_path).convert("RGB")

# Specify the organ to segment
# Available organs and their IDs:
# organ_to_class_dict = {
#     "appendix": 0,
#     "breast": 1,
#     "breast_luminal": 1,
#     "cardiac": 2,
#     "thyroid": 3,
#     "fetal": 4,
#     "kidney": 5,
#     "liver": 6,
#     "testicle": 7,
# }
organ_id = 4  # for fetal

# Preprocess the image and prepare inputs for the model
inputs = processor(images=image, return_tensors="pt")
inputs["organ_id"] = torch.tensor([organ_id])


# Perform inference
with torch.no_grad():
    outputs = model(**inputs, organ_id=organ_id)

# Post-process the output mask as needed
mask = processor.post_process_semantic_segmentation(
    outputs, 
    inputs, 
    threshold=0.7, 
    return_as_pil=True
)[0]

 Save the result
mask.save("output_mask.png")
```
*Note: The normalization values should be adjusted to match the ones used for training the model. Please refer to the model card on Hugging Face for more details.*

## Fine-tuning on a New Dataset

You can fine-tune US-FiLMUNet on your own dataset, using pretrained weighst publicy available at:
 - [US-FiLMUNet5](https://huggingface.co/AImageLab-Zip/US_FiLMUNet/resolve/main/model.safetensors) 
 - [US-FiLMUNet4](https://huggingface.co/AImageLab-Zip/US_FiLMUNet/resolve/main/unet_4_stages/model.safetensors)

### 1. Dataset Formatting

You need to structure your dataset in the following way. You should have a directory with your images, a directory with the corresponding segmentation masks, and a JSON file that for bounding boxes.

**Directory Structure:**

```
/path/to/your/datasets/
├── Dataset1
├── Dataset2
├── Your_Dataset
    ├── imgs/
    │   ├── image1.png
    │   ├── image2.png
    │   └── ...
    ├── masks/
    │   ├── image1.png
    │   ├── image2.png
    │   └── ...
    └── bboxes.json
```

**JSON file format:**

The `bboxes.json` file shall contain the bounding boxes coordinates of the masks, it can be generated using the provided `generate_bboxes.py` script.
```bash 
python generate_bboxes.py --dataset_dir /path/to/your/datasets/dataset --mask_ext png(or whatever)
```

Along with the bounding boxes the script will also generate the 
 - train.txt
 - val.txt
 - val_cls.txt (commonly the test file)

These files are mandatory as they contain the image names required for the dataset,so they are either creted through the script or you can create them manually



### 2. Running the Training Script

Once your dataset is ready, you can start the fine-tuning process by running the `main_segm.py` script.

Here is an example command:

```bash
python main_segm.py \ 
    --dataset-path /path/to/your/dataset \
    --dataset-type segmentation --dataset-size 512 --keep-aspect-ratio 1 \
    --film-start 0 --unet-depth 5 --freeze-image-encoder 0 \
    --learning-rate 1e-4 --batch-size 1 --epochs 5 --acc-grad 4 --sft 1 
```

You can find more details about the available arguments in the `main_segm.py` script.
Also make sure to fill the `utils/paths.py` with the path in which the weights are saved