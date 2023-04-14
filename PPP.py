#!/usr/bin/env python
# coding: utf-8

# # GPT-2 Fine-Tuning for the PPP project
# This is a simplified script for fine-tuning GPT2 using Hugging Face's Transformers library, PyTorch, and the "eli5" dataset from Hugging Face's datasets library.

# ### Setup and installing the needed libraries :

# In[16]:


pip install torch torchvision --user


# In[1]:


# The Transformers library provides a wide range of pre-trained models, tokenizers, and utilities for NLP tasks such as text classification, question-answering, and language generation.
get_ipython().system('pip install transformers')


# In[2]:


# The Datasets library provides access to a large collection of public datasets for NLP tasks.
get_ipython().system('pip install datasets')


# In[2]:


# Importing classes and functions from the Transformers library.
from transformers import GPT2LMHeadModel,  GPT2Tokenizer, GPT2Config
from transformers import AdamW, get_linear_schedule_with_warmup


# GPT2LMHeadModel: a pre-trained GPT-2 model for language generation and completion.
# GPT2Tokenizer: a tokenizer for the GPT-2 model
# GPT2Config: a configuration class for the GPT-2 model.
# AdamW: an optimizer for training neural networks with weight decay.
# get_linear_schedule_with_warmup: a function that generates a learning rate schedule with warmup for training neural networks.

# In[13]:


# Load the GPT tokenizer.
tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>')
# Instantiate a configuration for the model, but it's not really needed.
configuration = GPT2Config.from_pretrained('gpt2', output_hidden_states=False)
# Instantiate the pre-trained model.
model = GPT2LMHeadModel.from_pretrained("gpt2", config=configuration)

# Resizes the model's token embeddings Matrix (Matrix of tokens IDs) to match the size of the tokenizer's vocabulary.
model.resize_token_embeddings(len(tokenizer))


# In[22]:


import torch
# Tell pytorch to run this model on the GPU for faster and more efficient computation of deep learning models. [Change the runtime type]
device = torch.device("cuda")
model.cuda()


# In[10]:


model.device


# In[21]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    model = model.to(device)
except RuntimeError as e:
    print(f"Failed to move the model to device {device}: {e}")


# In[ ]:




