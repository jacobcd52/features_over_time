import torch as t
from nnsight import LanguageModel
import gc
from tqdm import tqdm

from config import DEBUG

if DEBUG:
    tracer_kwargs = {'scan' : True, 'validate' : True}
else:
    tracer_kwargs = {'scan' : False, 'validate' : False}


class MultiModelActivationBuffer:
    """
    Implements a buffer of activations from multiple models. The buffer stores activations from multiple models,
    yields them in batches, and refreshes them when the buffer is less than half full.
    """
    def __init__(self, 
                 data, # generator which yields text data
                 model_list, # list of LanguageModels from which to extract activations
                 submodule_list, # list of submodules from which to extract activations
                 d_submodule=None, # submodule dimension; if None, try to detect automatically
                 io='out', # can be 'in' or 'out'; whether to extract input or output activations
                 n_ctxs=3e4, # approximate number of contexts to store in the buffer
                 ctx_len=128, # length of each context
                 refresh_batch_size=512, # size of batches in which to process the data when adding to buffer
                 out_batch_size=8192, # size of batches in which to yield activations
                 device='cpu', # device on which to store the activations
                 remove_bos: bool = False,
                 ):
        
        if io not in ['in', 'out']:
            raise ValueError("io must be either 'in' or 'out'")

        self.n_models = len(model_list)
        if len(submodule_list) != self.n_models:
            raise ValueError("Length of submodule_list must match length of model_list")

        if d_submodule is None:
            try:
                if io == 'in':
                    d_submodule = submodule_list[0].in_features
                else:
                    d_submodule = submodule_list[0].out_features
            except:
                raise ValueError("d_submodule cannot be inferred and must be specified directly")
            
            # Verify all submodules have same dimension
            for submodule in submodule_list[1:]:
                try:
                    d = submodule.in_features if io == 'in' else submodule.out_features
                    if d != d_submodule:
                        raise ValueError("All submodules must have the same dimension")
                except:
                    raise ValueError("d_submodule cannot be inferred for all submodules")

        # Change: store activations with concatenated dimension
        self.activations = t.empty(0, self.n_models * d_submodule, device=device, dtype=model_list[0].dtype)
        self.read = t.zeros(0).bool()

        self.data = data
        self.model_list = model_list
        self.submodule_list = submodule_list
        self.d_submodule = d_submodule
        self.io = io
        self.n_ctxs = n_ctxs
        self.ctx_len = ctx_len
        self.activation_buffer_size = n_ctxs * ctx_len
        self.refresh_batch_size = refresh_batch_size
        self.out_batch_size = out_batch_size
        self.device = device
        self.remove_bos = remove_bos

    def refresh(self):
        gc.collect()
        t.cuda.empty_cache()
        self.activations = self.activations[~self.read]

        current_idx = len(self.activations)
        # Change: use concatenated dimension for new activations
        new_activations = t.empty(self.activation_buffer_size, self.n_models * self.d_submodule, 
                                device=self.device, dtype=self.model_list[0].dtype)

        new_activations[:len(self.activations)] = self.activations
        self.activations = new_activations

        while current_idx < self.activation_buffer_size:
            texts = self.text_batch()
            all_model_activations = []
            all_seq_lengths = []

            # Process the same text through each model
            for model, submodule in zip(self.model_list, self.submodule_list):
                with t.no_grad():
                    with model.trace(
                        texts,
                        **tracer_kwargs,
                        invoker_args={"truncation": True, "max_length": self.ctx_len},
                    ):
                        if self.io == "in":
                            hidden_states = submodule.inputs[0].save()
                        else:
                            hidden_states = submodule.output.save()
                        input = model.inputs.save()

                        submodule.output.stop()
                attn_mask = input.value[1]["attention_mask"]
                hidden_states = hidden_states.value
                if isinstance(hidden_states, tuple):
                    hidden_states = hidden_states[0]
                if self.remove_bos:
                    hidden_states = hidden_states[:, 1:, :]
                    attn_mask = attn_mask[:, 1:]
                hidden_states = hidden_states[attn_mask != 0]
                all_model_activations.append(hidden_states)
                all_seq_lengths.append(len(hidden_states))

            # Use the minimum sequence length across models
            min_seq_length = min(all_seq_lengths)
            remaining_space = self.activation_buffer_size - current_idx
            min_seq_length = min(min_seq_length, remaining_space)

            # Change: concatenate instead of stack activations
            concat_activations = t.cat([acts[:min_seq_length] for acts in all_model_activations], dim=1)
            self.activations[current_idx:current_idx + min_seq_length] = concat_activations.to(self.device)
            current_idx += min_seq_length

        self.read = t.zeros(len(self.activations), dtype=t.bool, device=self.device)
    
    def __iter__(self):
        return self

    def __next__(self):
        """
        Return a batch of activations with shape [batch, n_models, d]
        """
        with t.no_grad():
            # if buffer is less than half full, refresh
            if (~self.read).sum() < self.activation_buffer_size // 2:
                self.refresh()

            # return a batch
            unreads = (~self.read).nonzero().squeeze()
            idxs = unreads[t.randperm(len(unreads), device=unreads.device)[:self.out_batch_size]]
            self.read[idxs] = True
            return self.activations[idxs]
    
    def text_batch(self, batch_size=None):
        """
        Return a list of text
        """
        if batch_size is None:
            batch_size = self.refresh_batch_size
        try:
            return [
                next(self.data) for _ in range(batch_size)
            ]
        except StopIteration:
            raise StopIteration("End of data stream reached")
    
    def tokenized_batch(self, batch_size=None, model_idx=0):
        """
        Return a batch of tokenized inputs.
        """
        texts = self.text_batch(batch_size=batch_size)
        return self.model_list[model_idx].tokenizer(
            texts,
            return_tensors='pt',
            max_length=self.ctx_len,
            padding=True,
            truncation=True
        )

  
    @property
    def config(self):
        return {
            'n_models': self.n_models,
            'd_submodule': self.d_submodule,
            'io': self.io,
            'n_ctxs': self.n_ctxs,
            'ctx_len': self.ctx_len,
            'refresh_batch_size': self.refresh_batch_size,
            'out_batch_size': self.out_batch_size,
            'device': self.device
        }

    def close(self):
        """
        Close the text stream and the underlying compressed file.
        """
        self.text_stream.close()