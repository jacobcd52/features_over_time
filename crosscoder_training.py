#%%
from datasets import load_dataset
import torch as t

from nnsight import LanguageModel
from buffer import MultiModelActivationBuffer
from trainers.top_k import TopKTrainer, AutoEncoderTopK
from training import trainSAE

device = "cuda:0"
dtype = t.bfloat16

#%%
layer = 4
expansion = 8
num_tokens = int(200e6)
out_batch_size = 8192 * 2
n_init_batches = 10

submodule_list = []
model_list = []
for step in [1, 128, 256, 512, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 143000]:
    model = LanguageModel(
        "EleutherAI/pythia-70m", 
        revision=f"step{step}", 
        trust_remote_code=False, 
        device_map=device,
        torch_dtype=dtype,
        )
    for x in model.parameters():
        x.requires_grad = False
    model_list.append(model)
    submodule_list.append(model.gpt_neox.layers[layer])
    
activation_dim = 512
dictionary_size = expansion * activation_dim

dataset = load_dataset(
    'Skylion007/openwebtext', 
    split='train', 
    streaming=True,
    trust_remote_code=True
    )

class CustomData():
    def __init__(self, dataset):
        self.data = iter(dataset)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.data)['text']

data = CustomData(dataset)

buffer = MultiModelActivationBuffer(
    data=data,
    model_list=model_list,
    submodule_list=submodule_list,
    d_submodule=activation_dim, # output dimension of the model component
    n_ctxs=1024,  # you can set this higher or lower dependong on your available memory
    device=device,
    refresh_batch_size=512,
    out_batch_size=out_batch_size,
    rescale_acts=True,
    n_init_batches=n_init_batches,
    remove_bos=True
)  # buffer will yield batches of tensors of dimension = submodule's output dimension

#%%
trainer_cfg = {
    "trainer": TopKTrainer,
    "dict_class": AutoEncoderTopK,
    "activation_dim": activation_dim * len(model_list),
    "dict_size": dictionary_size,
    "device": device,
    "steps": num_tokens // out_batch_size,
    "k": 128,
    "layer": layer,
    "lm_name": "blah",
    "warmup_steps": 0,
}

# train the sparse autoencoder (SAE)
ae = trainSAE(
    data=buffer,  # you could also use another (i.e. pytorch dataloader) here instead of buffer
    trainer_configs=[trainer_cfg],
    steps=num_tokens // out_batch_size,
    autocast_dtype=dtype,
    use_wandb=True,
    wandb_project="features over time",
    log_steps=20,
    hf_repo_out="jacobcd52/features_over_time_narrow",
    save_dir="/root/features_over_time/checkpoints/",
)
# %%
