# [StarCoder2-15b](https://huggingface.co/bigcode/starcoder2-15b) Demonstration


```python
pip install git+https://github.com/huggingface/transformers.git
```

    huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
    To disable this warning, you can either:
    	- Avoid using `tokenizers` before the fork if possible
    	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)


    Collecting git+https://github.com/huggingface/transformers.git
      Cloning https://github.com/huggingface/transformers.git to /tmp/pip-req-build-pq5y57c2
      Running command git clone --filter=blob:none --quiet https://github.com/huggingface/transformers.git /tmp/pip-req-build-pq5y57c2
      Resolved https://github.com/huggingface/transformers.git to commit 3b6e95ec7fb08ad9bef4890bcc6969d68cc70ddb
      Installing build dependencies ... [?25ldone
    [?25h  Getting requirements to build wheel ... [?25ldone
    [?25h  Preparing metadata (pyproject.toml) ... [?25ldone
    [?25hRequirement already satisfied: filelock in /usr/local/lib/python3.10/dist-packages (from transformers==4.39.0.dev0) (3.9.0)
    Requirement already satisfied: huggingface-hub<1.0,>=0.19.3 in /usr/local/lib/python3.10/dist-packages (from transformers==4.39.0.dev0) (0.21.4)
    Requirement already satisfied: numpy>=1.17 in /usr/local/lib/python3.10/dist-packages (from transformers==4.39.0.dev0) (1.24.1)
    Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.10/dist-packages (from transformers==4.39.0.dev0) (23.2)
    Requirement already satisfied: pyyaml>=5.1 in /usr/local/lib/python3.10/dist-packages (from transformers==4.39.0.dev0) (6.0.1)
    Requirement already satisfied: regex!=2019.12.17 in /usr/local/lib/python3.10/dist-packages (from transformers==4.39.0.dev0) (2023.12.25)
    Requirement already satisfied: requests in /usr/local/lib/python3.10/dist-packages (from transformers==4.39.0.dev0) (2.31.0)
    Requirement already satisfied: tokenizers<0.19,>=0.14 in /usr/local/lib/python3.10/dist-packages (from transformers==4.39.0.dev0) (0.15.2)
    Requirement already satisfied: safetensors>=0.4.1 in /usr/local/lib/python3.10/dist-packages (from transformers==4.39.0.dev0) (0.4.2)
    Requirement already satisfied: tqdm>=4.27 in /usr/local/lib/python3.10/dist-packages (from transformers==4.39.0.dev0) (4.66.2)
    Requirement already satisfied: fsspec>=2023.5.0 in /usr/local/lib/python3.10/dist-packages (from huggingface-hub<1.0,>=0.19.3->transformers==4.39.0.dev0) (2024.2.0)
    Requirement already satisfied: typing-extensions>=3.7.4.3 in /usr/local/lib/python3.10/dist-packages (from huggingface-hub<1.0,>=0.19.3->transformers==4.39.0.dev0) (4.4.0)
    Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests->transformers==4.39.0.dev0) (2.1.1)
    Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests->transformers==4.39.0.dev0) (3.4)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests->transformers==4.39.0.dev0) (1.26.13)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests->transformers==4.39.0.dev0) (2022.12.7)
    [33mWARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv[0m[33m
    [0m
    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m A new release of pip is available: [0m[31;49m23.3.1[0m[39;49m -> [0m[32;49m24.0[0m
    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m To update, run: [0m[32;49mpython -m pip install --upgrade pip[0m
    Note: you may need to restart the kernel to use updated packages.



```python
from transformers import AutoModelForCausalLM, AutoTokenizer
```


```python
checkpoint = "bigcode/starcoder2-15b"
device = "cuda" # for GPU usage or "cpu" for CPU usage
```


```python
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# for multiple GPUs install accelerate and do `model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto")`
model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
```


    tokenizer_config.json:   0%|          | 0.00/7.88k [00:00<?, ?B/s]



    vocab.json:   0%|          | 0.00/777k [00:00<?, ?B/s]



    merges.txt:   0%|          | 0.00/442k [00:00<?, ?B/s]



    tokenizer.json:   0%|          | 0.00/2.06M [00:00<?, ?B/s]



    special_tokens_map.json:   0%|          | 0.00/958 [00:00<?, ?B/s]



    config.json:   0%|          | 0.00/803 [00:00<?, ?B/s]



    model.safetensors.index.json:   0%|          | 0.00/52.1k [00:00<?, ?B/s]



    Downloading shards:   0%|          | 0/14 [00:00<?, ?it/s]



    model-00001-of-00014.safetensors:   0%|          | 0.00/4.61G [00:00<?, ?B/s]



    model-00002-of-00014.safetensors:   0%|          | 0.00/4.61G [00:00<?, ?B/s]



    model-00003-of-00014.safetensors:   0%|          | 0.00/4.61G [00:00<?, ?B/s]



    model-00004-of-00014.safetensors:   0%|          | 0.00/4.61G [00:00<?, ?B/s]



    model-00005-of-00014.safetensors:   0%|          | 0.00/4.61G [00:00<?, ?B/s]



    model-00006-of-00014.safetensors:   0%|          | 0.00/4.61G [00:00<?, ?B/s]



    model-00007-of-00014.safetensors:   0%|          | 0.00/4.61G [00:00<?, ?B/s]



    model-00008-of-00014.safetensors:   0%|          | 0.00/4.61G [00:00<?, ?B/s]



    model-00009-of-00014.safetensors:   0%|          | 0.00/4.61G [00:00<?, ?B/s]



    model-00010-of-00014.safetensors:   0%|          | 0.00/4.61G [00:00<?, ?B/s]



    model-00011-of-00014.safetensors:   0%|          | 0.00/4.61G [00:00<?, ?B/s]



    model-00012-of-00014.safetensors:   0%|          | 0.00/4.61G [00:00<?, ?B/s]



    model-00013-of-00014.safetensors:   0%|          | 0.00/4.61G [00:00<?, ?B/s]



    model-00014-of-00014.safetensors:   0%|          | 0.00/3.95G [00:00<?, ?B/s]



    Loading checkpoint shards:   0%|          | 0/14 [00:00<?, ?it/s]



    generation_config.json:   0%|          | 0.00/124 [00:00<?, ?B/s]



```python
inputs = tokenizer.encode("import React from 'react';\nimport { Link } from 'react-router-dom';\nexport default", return_tensors="pt").to(device)
```


```python
inputs = tokenizer.encode("Create one example of the completed code with one exported function: import React from 'react';\nimport { Link } from 'react-router-dom';\nexport default", return_tensors="pt").to(device)
```


```python
inputs = tokenizer.encode("Create a python script that loads a StarCoder2-15b from hugging face and runs inferences", return_tensors="pt").to(device)
```


```python
outputs = model.generate(inputs, max_length=2000)
print(tokenizer.decode(outputs[0]))
```

    The attention mask and the pad token id were not set. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
    Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.


    Create a python script that loads a StarCoder2-15b from hugging face and runs inferences on a few examples.
    
    ```python
    from transformers import StarCoder2ForCausalLM
    
    model = StarCoder2ForCausalLM.from_pretrained("bigscience/starcoder2-15b")
    
    # Run inference on a few examples
    inputs = [
        "def foo(x):",
        "def bar(x):",
        "def baz(x):",
    ]
    
    outputs = model.generate(
        inputs,
        max_new_tokens=100,
        do_sample=True,
        top_p=0.95,
        top_k=50,
        temperature=0.7,
        repetition_penalty=1.1,
        eos_token_id=1,
    )
    
    for input, output in zip(inputs, outputs):
        print(f"Input: {input}")
        print(f"Output: {output}")
        print()
    ```
    
    ## StarCoder2-15b-instruct
    
    StarCoder2-15b-instruct is a StarCoder2-15b model fine-tuned on a large dataset of 100M instruction-following examples.
    
    ### Usage
    
    Create a python script that loads a StarCoder2-15b-instruct from hugging face and runs inferences on a few examples.
    
    ```python
    from transformers import StarCoder2ForCausalLM
    
    model = StarCoder2ForCausalLM.from_pretrained("bigscience/starcoder2-15b-instruct")
    
    # Run inference on a few examples
    inputs = [
        "def foo(x):",
        "def bar(x):",
        "def baz(x):",
    ]
    
    outputs = model.generate(
        inputs,
        max_new_tokens=100,
        do_sample=True,
        top_p=0.95,
        top_k=50,
        temperature=0.7,
        repetition_penalty=1.1,
        eos_token_id=1,
    )
    
    for input, output in zip(inputs, outputs):
        print(f"Input: {input}")
        print(f"Output: {output}")
        print()
    ```
    
    ## StarCoder2-15b-instruct-code-to-code
    
    StarCoder2-15b-instruct-code-to-code is a StarCoder2-15b model fine-tuned on a large dataset of 100M instruction-following examples.
    
    ### Usage
    
    Create a python script that loads a StarCoder2-15b-instruct-code-to-code from hugging face and runs inferences on a few examples.
    
    ```python
    from transformers import StarCoder2ForCausalLM
    
    model = StarCoder2ForCausalLM.from_pretrained("bigscience/starcoder2-15b-instruct-code-to-code")
    
    # Run inference on a few examples
    inputs = [
        "def foo(x):",
        "def bar(x):",
        "def baz(x):",
    ]
    
    outputs = model.generate(
        inputs,
        max_new_tokens=100,
        do_sample=True,
        top_p=0.95,
        top_k=50,
        temperature=0.7,
        repetition_penalty=1.1,
        eos_token_id=1,
    )
    
    for input, output in zip(inputs, outputs):
        print(f"Input: {input}")
        print(f"Output: {output}")
        print()
    ```
    
    ## StarCoder2-15b-instruct-code-to-text
    
    StarCoder2-15b-instruct-code-to-text is a StarCoder2-15b model fine-tuned on a large dataset of 100M instruction-following examples.
    
    ### Usage
    
    Create a python script that loads a StarCoder2-15b-instruct-code-to-text from hugging face and runs inferences on a few examples.
    
    ```python
    from transformers import StarCoder2ForCausalLM
    
    model = StarCoder2ForCausalLM.from_pretrained("bigscience/starcoder2-15b-instruct-code-to-text")
    
    # Run inference on a few examples
    inputs = [
        "def foo(x):",
        "def bar(x):",
        "def baz(x):",
    ]
    
    outputs = model.generate(
        inputs,
        max_new_tokens=100,
        do_sample=True,
        top_p=0.95,
        top_k=50,
        temperature=0.7,
        repetition_penalty=1.1,
        eos_token_id=1,
    )
    
    for input, output in zip(inputs, outputs):
        print(f"Input: {input}")
        print(f"Output: {output}")
        print()
    ```
    
    ## StarCoder2-15b-instruct-text-to-code
    
    StarCoder2-15b-instruct-text-to-code is a StarCoder2-15b model fine-tuned on a large dataset of 100M instruction-following examples.
    
    ### Usage
    
    Create a python script that loads a StarCoder2-15b-instruct-text-to-code from hugging face and runs inferences on a few examples.
    
    ```python
    from transformers import StarCoder2ForCausalLM
    
    model = StarCoder2ForCausalLM.from_pretrained("bigscience/starcoder2-15b-instruct-text-to-code")
    
    # Run inference on a few examples
    inputs = [
        "def foo(x):",
        "def bar(x):",
        "def baz(x):",
    ]
    
    outputs = model.generate(
        inputs,
        max_new_tokens=100,
        do_sample=True,
        top_p=0.95,
        top_k=50,
        temperature=0.7,
        repetition_penalty=1.1,
        eos_token_id=1,
    )
    
    for input, output in zip(inputs, outputs):
        print(f"Input: {input}")
        print(f"Output: {output}")
        print()
    ```
    
    ## StarCoder2-15b-instruct-text-to-text
    
    StarCoder2-15b-instruct-text-to-text is a StarCoder2-15b model fine-tuned on a large dataset of 100M instruction-following examples.
    
    ### Usage
    
    Create a python script that loads a StarCoder2-15b-instruct-text-to-text from hugging face and runs inferences on a few examples.
    
    ```python
    from transformers import StarCoder2ForCausalLM
    
    model = StarCoder2ForCausalLM.from_pretrained("bigscience/starcoder2-15b-instruct-text-to-text")
    
    # Run inference on a few examples
    inputs = [
        "def foo(x):",
        "def bar(x):",
        "def baz(x):",
    ]
    
    outputs = model.generate(
        inputs,
        max_new_tokens=100,
        do_sample=True,
        top_p=0.95,
        top_k=50,
        temperature=0.7,
        repetition_penalty=1.1,
        eos_token_id=1,
    )
    
    for input, output in zip(inputs, outputs):
        print(f"Input: {input}")
        print(f"Output: {output}")
        print()
    ```
    
    ## StarCoder2-15b-instruct-code-to-code-code-to-text
    
    StarCoder2-15b-instruct-code-to-code-code-to-text is a StarCoder2-15b model fine-tuned on a large dataset of 100M instruction-following examples.
    
    ### Usage
    
    Create a python script that loads a StarCoder2-15b-instruct-code-to-code-code-to-text from hugging face and runs inferences on a few examples.
    
    ```python
    from transformers import StarCoder2ForCausalLM
    
    model = StarCoder2ForCausalLM.from_pretrained("bigscience/starcoder2-15b-instruct-code-to-code-code-to-text")
    
    # Run inference on a few examples
    inputs = [
        "def foo(x):",
        "def bar(x):",
        "def baz(x):",
    ]
    
    outputs = model.generate(
        inputs,
       



```python

```
