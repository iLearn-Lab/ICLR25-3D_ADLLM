# 3d_aff

Training repository of 3D Affordance LLM

# Getting Started

## Installation

In order to install the dependencies, you can use the following commands:

```bash
conda create -n 3daff
conda activate 3daff
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

## Running Configuration

The configurations for running are stored in `configs/` directory. It contains four fields, which are
| Field          | Description                        |
| -------------- | ---------------------------------- |
| run            | Training hyperparameters           |
| model          | Model configurations               |
| train_datasets | Training datasets                  |
| eval_datasets  | Evaluation datasets and evaluators |

Explanation for some keys.

- Field `run`

  - `evaluate`: Set to `True` for evaluation only.

- Field `model`

  - Use `arch` and `model_type` to select a specific model.
  - You can also override any corresponding entry in original model configuration file here.

- Field `train_datasets`

  - `type ` indicates the data loader to use

  ```yaml
  train_datasets:
    dataset_name:
      type: affordance
      sample_ratio: 10
  ```

- Field `eval_datasets`

  - `eval_type ` indicates the evaluator to use

  ```yaml
  train_datasets:
    dataset_name:
      type: affordance
      eval_type: affordance_acc
  ```


## Training

For multi-gpu training, you can use the following command:
```bash
CUDA_VISIBLE_DEVICES=0,1 TOKENIZERS_PARALLELISM=true  torchrun --master_port 11453 --nproc_per_node=2 train.py --cfg-path configs/phi_train/phi_train.yaml

```
For single-gpu training, you can use the following command:
```bash
python train.py --cfg-path configs/T5_finetune.yaml
```
For deepspeed training: Please refer to [HuggingFace Accelerator Document](https://huggingface.co/docs/accelerate/v0.22.0/en/usage_guides/deepspeed) for more details.

Here is an example:

```bash
accelerate launch --config_file ds_config.yaml train.py --cfg-path configs/test.yaml
```
## Affordance Model Test
```bash
# Single GPU training
CUDA_VISIBLE_DEVICES=1 python train.py --cfg-path configs/point_affordance_openad.yaml 

# Multi Gpu Training
CUDA_VISIBLE_DEVICES=0,1 TOKENIZERS_PARALLELISM=true  torchrun --master_port 11453 --nproc_per_node=2 train.py --cfg-path configs/point_affordance_DGCNN.yaml > outputs/dgcnn_llm/log/train_dgcnn_batch_4_lr_3e-4_iteration_5000_seed_1.log 2>&1 &
```

## Affordance Model 
```bash
# deepspeed training
# train for model identify
accelerate launch --config_file ds_config.yaml train.py --cfg-path configs/point_affordance_DGCNN_pretrained.yaml > outputs/dgcnn_llm/log/train_dgcnn_batch_4_lr_3e-4_iteration_5000_seed_1.log 2>&1 &

```

## Adding Custom Models

### Model

You can follow the instructions below to create any custom models.

1. Create new model class in `models/`  directory. Models should inherit from `nn.Model`. 
2. The model should have a **Decorator** `@registry.register_model("model_name")`, which indicates the **model name**.
3. The model should have a **class variable** named `PRETRAINED_MODEL_CONFIG_DICT`, which is a python dictionary, indicating all optional **model type** and their corresponding configuration file paths.
4. The model should contain 2 special method: `load_from_pretrained` for loading pretrained checkpoint (you can just copy from the example below). Class method `from_config` for reading model configuration file.
5. The `forward ` method should return a dictionary containing key `loss`. For inference, you may implement a different method `generate`. 
6. The `forward ` method receives a dictionary provided by data loader.
7. Import your model in `models/__init__.py`

Here is an example for creating new model.

```python
@registry.register_model("linear_vicuna_instruct")
class LinearVicuna(Basemodel):
    PRETRAINED_MODEL_CONFIG_DICT = {
        "vicuna7b": "configs/models/linear_vicuna.yaml",
        ...
    }
    def __init__(
            self,
            param1,
            param2,
            ...,
            **kwargs
    ):
        super().__init__()
        
    def forward(self, samples_dict):
        # ...
        return {"loss": loss}
        
    def load_from_pretrained(self, url_or_filename):
        if os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location="cpu")
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        state_dict = checkpoint["model"]
        msg = self.load_state_dict(state_dict, strict=False)
        logging.info("load checkpoint from %s" % url_or_filename)
        return msg
        
    @classmethod
    def from_config(cls, cfg):
        param1 = cfg.get("param1", None)
        param2 = cfg.get("param2", None)

        model = cls(
            param1=param1,
			      param2=param2
        )

        model.load_checkpoint_from_config(cfg)
        return model
```

### Configuration

After creating the model class, you can follow the instructions below to create corresponding configuration file.

1. Key `model` indicates the configuration for model. Any child key under key `model` can be read using `cfg.get()`  in model's `from_config` method.

And here is an example for creating corresponding configuration file.

```yaml
model:
  param1: 123
  param2: 234

```

### Training

Each training configuration file also contains a `model` key. Add two special key under key `model`: 

1. `arch` for **model name** （the name you defined in **decorator**) 
2. `model_type` for different model configuration (the configuration dictionary you defined in ``PRETRAINED_MODEL_CONFIG_DICT``)

You can also override any key in original configuration file here.

```yaml
model:
  arch: comm_vicuna_clip_adapter
  model_type: vicuna7b
  
  load_pretrained: True
  pretrained: "path/to/checkpoint.pth"
  
run:
  ....
```

### Inference
<!-- 3D_ADLLM，我没有定义preprocess -->
For inference, you can use `load_model_and_preprocess` method along with your **model name** and **model type** to load the model. Then just call the implemented `generate` method.

```python
from models import load_model_and_preprocess

model, processor, _ = load_model_and_preprocess(
    name=args.model_name,
    model_type=args.model_arch,
    is_eval=True,
    device="cuda:0",
)
model.load_from_pretrained(args.ckpt)

```

## Adding Custom Data Loader

### Dataloader

You can follow the instructions below to create any custom data loaders.

1. Create new data loader class in `dataset/`  directory. data loader should inherit from `torch.utils.data.Dataset`. 
2. Implement `__len__` method and `__getitem__` method. 

```python
class VQADataset(Dataset):
    def __init__(self):
        self.annotation = read_from_file()

    def __len__(self):
        return len(self.annotation)
        
    def __getitem__(self, index):
        return {
            "image": image,
            "question": question,
            "answer": answers
        }
```

After creating your data loader class, register it in `dataset/builder.py`.

1. Add a **decorator** `@registry.register_builder(dataloader_name) `, which defines its name.
2. Define what datasets can be support by this loader.

```python
@registry.register_builder("vqa") 
def build_vqa_dataset(name):
    cfg_path = "configs/datasets/vqa_datasets.yaml"
    all_cfg = OmegaConf.load(cfg_path)
    cfg = all_cfg.get(name, None)
    assert cfg is not None, f"Dataset {name} not found in {cfg_path}"
    
    dataset = VQADataset(....)
    return dataset
```

Here is an example to use your custom data loader.

```yaml
train_datasets:
  dataset_name:
    type: vqa
```
### Datsets

Datasets configurations are stored in `configs/datasets`. An example of LRV dataset is shown below:
<!-- 这里主要看数据Dataset的定义情况 -->
<!-- 这里如果前面我们有定义相应的processor,那么这里可以设置-->
```yaml
LRV:
  ann_path: "path/to/LRV.json" # path to the annotation json file
  vis_processor: "blip2_image_train" # the name of the visual processor registered in folder `processors`
```
<!-- Else -->
```yaml
3D_Affordance_train:
  ann_path: "/workspace/Dataset/process_data_for_affllm/final_train_data_full_shape_mask.pkl" # path to the data
```
For easy switching between servers, the path in the dataset configuration can be an absolute path or a relative path with the root set in `common/registry.py`. You can set the root by `registry.root = "path/to/root"`.

## Adding Custom Evaluator

You can follow the instructions below to create any custom evaluators.

1. Create new evaluator class in `evaluator`  directory. 
2. Add a **decorator** `@registry.register_evaluator(name) `, which defines its name.
3. Register in `evaluator/__init__.py`.

# Acknowledgement

[LAVIS](https://github.com/salesforce/LAVIS): This repository is modified from LAVIS.