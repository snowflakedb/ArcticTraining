# SwiftKV + MOBA Integration

Clean integration of MOBA (Mixture of Block Attention) with SwiftKV LLaMA models using inheritance.

## 🎯 **Clean Inheritance Approach**

Instead of modifying the original SwiftKV implementation, this approach:

1. **Inherits from SwiftKV**: `LlamaSwiftKVMoBAConfig` extends `LlamaSwiftKVConfig`
2. **Registers MOBA Attention**: Adds MOBA to `ALL_ATTENTION_FUNCTIONS`  
3. **Zero Modifications**: Original SwiftKV code remains untouched
4. **Clean Separation**: MOBA code lives in `moba_attention/models/swiftkv_llama/`

## 📁 **Directory Structure**

```
moba_attention/
├── models/
│   └── swiftkv_llama/           # ← New: SwiftKV + MOBA integration
│       ├── __init__.py          # Main interface
│       ├── configuration_swiftkv_moba.py  # Inherits from SwiftKV config  
│       ├── moba_attention.py    # MOBA attention registration
│       ├── example.py           # Usage examples
│       └── README.md           # This file
└── moba/                       # Original MOBA implementation
    ├── wrapper.py
    ├── moba_efficient.py
    └── ...

swiftkv/                        # ← Original: Unchanged!
├── models/
│   └── llama/
│       ├── configuration_llama_swiftkv.py  # Original config
│       ├── modeling_llama_swiftkv.py       # Original model
│       └── __init__.py                     # Original interface
```

## 🚀 **Quick Usage**

### Simple One-Line Setup

```python
from ArcticTraining.projects.moba_attention.models.swiftkv_llama import create_swiftkv_moba_model

# Create model with both optimizations
model = create_swiftkv_moba_model(
    "meta-llama/Llama-2-7b-hf",
    swiftkv=True,
    num_key_value_layers=16,      # SwiftKV: full KV for first 16 layers
    key_value_group_size=2,       # SwiftKV: share KV across 2-layer groups
    moba_chunk_size=4096,         # MOBA: 4K token chunks  
    moba_topk=8,                  # MOBA: top-8 chunk selection
    torch_dtype=torch.float16,
    device_map="auto"
)
```

### Inheritance Example

```python
from ArcticTraining.projects.moba_attention.models.swiftkv_llama import (
    LlamaSwiftKVMoBAConfig,
    LlamaSwiftKVMoBAForCausalLM,
    register_llama_swiftkv_moba
)

# Register models
register_llama_swiftkv_moba()

# Create config that inherits from SwiftKV
config = LlamaSwiftKVMoBAConfig(
    # SwiftKV parameters (inherited)
    swiftkv=True,
    num_key_value_layers=16,
    key_value_group_size=2,
    
    # MOBA parameters (new)
    moba_chunk_size=4096,
    moba_topk=8,
)

# Specialized SwiftKV + MOBA model!
model = LlamaSwiftKVMoBAForCausalLM(config)
```

### Direct Model Creation

```python
from ArcticTraining.projects.moba_attention.models.swiftkv_llama import LlamaSwiftKVMoBAForCausalLM

# Load from pretrained with MOBA built-in
model = LlamaSwiftKVMoBAForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    moba_chunk_size=4096,
    moba_topk=8,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Update MOBA config dynamically
model.update_moba_config(chunk_size=8192, topk=16)
```

## 🏗️ **Architecture**

### Class Hierarchy

```python
# Configuration inheritance
LlamaConfig                     # Base transformers config
    ↓
LlamaSwiftKVConfig             # Adds SwiftKV parameters  
    ↓
LlamaSwiftKVMoBAConfig         # Adds MOBA parameters (inherits everything)

# Model inheritance  
LlamaForCausalLM               # Base transformers model
    ↓
LlamaSwiftKVForCausalLM        # Adds SwiftKV functionality
    ↓
LlamaSwiftKVMoBAForCausalLM    # Adds MOBA functionality (inherits everything)
```

### How It Works

1. **Config Inheritance**: `LlamaSwiftKVMoBAConfig` gets all SwiftKV functionality automatically
2. **Model Inheritance**: `LlamaSwiftKVMoBAForCausalLM` inherits from `LlamaSwiftKVForCausalLM`
3. **Built-in MOBA**: MOBA registration and setup happens automatically in the model's `__init__`
4. **No Patching**: Everything is handled through clean inheritance - no runtime modifications needed

## ✅ **Benefits**

- **🔒 Zero Impact**: SwiftKV code completely unchanged
- **🧬 True Inheritance**: Gets all SwiftKV features automatically  
- **🔌 Pluggable**: MOBA can be enabled/disabled on any SwiftKV model
- **🛠️ Maintainable**: SwiftKV improvements automatically benefit MOBA
- **📦 Modular**: MOBA code isolated in separate module
- **🏷️ Semantic**: Clear model types (`llama_swiftkv` vs `llama_swiftkv_moba`)

## 🧪 **Running Examples**

```bash
cd ArcticTraining/projects/moba_attention/models/swiftkv_llama
python example.py
```

Example output:
```
=== Inheritance Approach Example ===
Config type: LlamaSwiftKVMoBAConfig
Inherits from: LlamaSwiftKVConfig
Model type: llama_swiftkv_moba
SwiftKV params: layers=16, group_size=1
MOBA params: chunk_size=4096, topk=8
Model class: LlamaSwiftKVMoBAForCausalLM
Model uses SwiftKV implementation with MOBA attention: moba
```

## 🔍 **Implementation Details**

### Clean Inheritance

```python
# Config inheritance
class LlamaSwiftKVMoBAConfig(LlamaSwiftKVConfig):
    model_type = "llama_swiftkv_moba"
    
    def __init__(self, moba_chunk_size=4096, moba_topk=8, **kwargs):
        super().__init__(**kwargs)  # Gets ALL SwiftKV functionality
        self.moba_chunk_size = moba_chunk_size
        self.moba_topk = moba_topk

# Model inheritance
class LlamaSwiftKVMoBAForCausalLM(LlamaSwiftKVForCausalLM):
    config_class = LlamaSwiftKVMoBAConfig
    
    def __init__(self, config):
        register_moba_attention()  # Register MOBA
        config._attn_implementation = "moba"  # Enable MOBA
        super().__init__(config)  # Initialize SwiftKV model
```

### Registration

```python
def register_llama_swiftkv_moba():
    register_llama_swiftkv()  # Register base SwiftKV models first
    
    # Register our specialized model class
    AutoConfig.register("llama_swiftkv_moba", LlamaSwiftKVMoBAConfig)
    AutoModel.register(LlamaSwiftKVMoBAConfig, LlamaSwiftKVModel)
    AutoModelForCausalLM.register(LlamaSwiftKVMoBAConfig, LlamaSwiftKVMoBAForCausalLM)  # Our class!
    
    register_moba_attention()  # Register MOBA in ALL_ATTENTION_FUNCTIONS
```
