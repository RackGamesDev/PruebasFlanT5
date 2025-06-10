#SCRIPT AJENO

#!pip install -q accelerate==0.21.0 peft==0.4.0 bitsandbytes==0.40.2 transformers==4.31.0 trl==0.4.7 xformers
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, pipeline, TrainingArguments
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, PeftModel
from trl import SFTTrainer


# Definimos los paramétros para bitsandbytes
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False,
)

# Leemos el modelo pre-entrenado el modelo LLAMA2-7b-chat
model = AutoModelForCausalLM.from_pretrained(
    "NousResearch/Llama-2-7b-chat-hf",
    quantization_config=bnb_config,
    device_map={"": 0},
    # low_cpu_mem_usage=True # Reduccion del consumo de cpu y memoria al leer el modelo
)

model.config.use_cache = False
model.config.pretraining_tp = 1 # Un valor distinto de 1 activará el cálculo más preciso pero más lento de las capas lineales



# Leemos el tokenizador
tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-chat-hf", trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training


# Creamos un pipeline para la tokenización y generación del texto
llama2_pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=500)

# Leemos el conjunto de datos de Google Drive
dataset_path = "cachedataset/llama-2-fine-tuning-datset.txt"

with open(dataset_path, 'r', encoding='utf-8') as f:
  examples = f.read().splitlines() # De esta forma no sale el \n



ds = Dataset.from_dict({"text": examples})

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )
print_trainable_parameters(model)


# Definición de la configuración de LoRA
lora_config = LoraConfig(
                 r = 16, # Dimensión de las matrices
                 lora_alpha = 16, # LoRA scaling factor
                 lora_dropout = 0.05, # Regularización
                 bias="none",
                 task_type="CAUSAL_LM" # Tipo de tarea/modelo al que aplicarlo
                 # target_modules=["q", "k", "v"], # Los módulos (ej. Attention Heads) donde aplicar las matrices
)

# Aplicamos la configuración al modelo
model_peft = get_peft_model(model, lora_config)

# Mostramos el número de parámetros que se van a entrenar
model_peft.print_trainable_parameters()

# Directorio de salida donde se almacenarán las predicciones del modelo y los puntos de control
output_dir = "prod/llama/llama2-7b-chat-peft"

# Número de epochs de entrenamiento
num_train_epochs = 5

# Habilitar entrenamiento fp16/bf16 (establecer bf16 en True con un A100)
fp16 = False
bf16 = False

# Tamaño del lote por GPU para el entrenamiento
per_device_train_batch_size = 4

# Tamaño del lote por GPU para la evaluación
per_device_eval_batch_size = 4

# Número de pasos de actualización para acumular los gradientes
gradient_accumulation_steps = 1

# Habilitar el registro de puntos de control de gradientes
gradient_checkpointing = True

# Norma máxima del gradiente (recorte del gradiente)
max_grad_norm = 0.3

# Tasa de aprendizaje inicial (optimizador AdamW)
learning_rate = 2e-4

# Decaimiento de pesos a aplicar a todas las capas excepto a los pesos de sesgo/LayerNorm
weight_decay = 0.001

# Optimizador a usar
optim = "paged_adamw_32bit"

# Programación de la tasa de aprendizaje
lr_scheduler_type = "cosine"

# Número de pasos de entrenamiento (anula num_train_epochs)
max_steps = -1

# Proporción de pasos para un calentamiento lineal (de 0 a tasa de aprendizaje)
warmup_ratio = 0.03

# Agrupar secuencias en lotes del mismo tamaño
# Ahorra memoria y acelera considerablemente el entrenamiento
group_by_length = True

# Guardar punto de control cada X pasos de actualización
save_steps = 0

# Registrar cada X pasos de actualización
logging_steps = 25



# Set training parameters
training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    fp16=fp16,
    bf16=bf16,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type,
    remove_unused_columns=False,
    save_strategy="epoch",
    save_total_limit=2
)

# Creamos la instancia de entrenamiento
trainer = SFTTrainer(
    model=model,
    train_dataset=ds,
    peft_config=lora_config,
    dataset_text_field="text",
    max_seq_length=None, # Cuando es None, el max_seq_len vendrá determinado por la secuencia más larga de un lote
    tokenizer=tokenizer,
    args=training_arguments,
    packing=False, # Empaquetar múltiples ejemplos cortos en la misma secuencia de entrada para aumentar la eficiencia
)

trainer.train()

tokenizer.save_pretrained(f"{output_dir}/tokenizer")


# Creamos un pipeline para la tokenización y generación del texto
llama2_ft_pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=500)



model_name = "NousResearch/Llama-2-7b-chat-hf"
adapters_name = "prod/llama/llama2-7b-chat-peft"

print(f"Cargando el modelo: '{model_name}' en memoria...")

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    #load_in_4bit=True,
    torch_dtype=torch.bfloat16,
    device_map={"": 0}
)

model = PeftModel.from_pretrained(model, adapters_name)
model = model.merge_and_unload()

print(f"El modelo: '{model_name}' ha sido cargado correctamente")

# Leemos el tokenizador
tokenizer = AutoTokenizer.from_pretrained("prod/llama/llama2-7b-chat-peft/tokenizer", trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training

# Creamos un pipeline para la tokenización y generación del texto
llama2_ft_pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=500)

print("TODO LISTO:")
prompt = ""
while prompt != 'SALIR':
    prompt = input('X: ')
    promptPreparado = "<s>[INST] <<SYS>>" + prompt + "<</SYS>>[/INST]"
    output = llama2_ft_pipe(promptPreparado)
    print("Y: " + output[0]['generated_text'].replace("[/INST]", "[/INST]\n\n"))








