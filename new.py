
#First of all, note that there are 2 ways to load a base model with its adapter weights:
from peft import PeftModel, PeftConfig, AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM

# let's say you fine-tuned OPT using PEFT

# method 1: separately
base_model_id = "facebook/opt-350m"
adapter_id = "ybelkada/opt-350m-lora"
base_model = AutoModelForCausalLM.from_pretrained(base_model_id)
base_with_adapters_model = PeftModel.from_pretrained(base_model, adapter_id)

# method 2: conveniently with the AutoPeftModelForCausalLM class
base_with_adapters_model = AutoPeftModelForCausalLM.from_pretrained("ybelkada/opt-350m-lora")

# now we just have a regular AutoModelForCausalLM Transformers model
model = base_with_adapters_model.merge_and_unload()


# next, we could apply PEFT again by adding another adapter
from peft import get_peft_model, LoraConfig, TaskType

lora_config = LoraConfig(
    r=16,
    target_modules=["q_proj", "v_proj"],
    task_type=TaskType.CAUSAL_LM,
    lora_alpha=32,
    lora_dropout=0.05
)

base_model_with_new_adapter = get_peft_model(model, lora_config)
base_model_with_new_adapter.print_trainable_parameters()

