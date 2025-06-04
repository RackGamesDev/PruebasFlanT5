#PAQUETES NECESARIOS
#!pip install transformers
#!pip install sentencepiece
#!pip install accelerate
#!pip install datasets
#!pip install evaluate
#!pip install rouge_score
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset, concatenate_datasets


#IMPORTAR EL MODELO BASE
#tokenizer_T5 = T5Tokenizer.from_pretrained("google/flan-t5-small")
tokenizer_T5 = T5Tokenizer.from_pretrained("google/flan-t5-large")
#model_T5 = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small", device_map="auto")
model_T5 = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large", device_map="auto", cache_dir="./cachemodel")


#DESCARGAR EL DATASET PARA ENTRENAR Y PREPARARLO
ds = load_dataset("mlsum", 'es', cache_dir="./cachedataset")

NUM_EJ_TRAIN = 1500
NUM_EJ_VAL = 500
NUM_EJ_TEST = 200
ds['train'] = ds['train'].select(range(NUM_EJ_TRAIN))
ds['validation'] = ds['validation'].select(range(NUM_EJ_VAL))
ds['test'] = ds['test'].select(range(NUM_EJ_TEST))

def parse_dataset(ejemplo):
    """Procesa los ejemplos para adaptarlos a la plantilla"""
    return {"prompt": f"Resumir el siguiente articulo: {ejemplo['text']}"}
ds["train"] = ds["train"].map(parse_dataset)
ds["validation"] = ds["validation"].map(parse_dataset)
ds["test"] = ds["test"].map(parse_dataset)
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")

prompts_tokens = concatenate_datasets([ds["train"], ds["validation"], ds["test"]]).map(lambda x: tokenizer(x["prompt"], truncation=True), batched=True)
max_token_len = max([len(x) for x in prompts_tokens["input_ids"]])
print(f"Maximo tamagno de prompt: {max_token_len}")

completions_tokens = concatenate_datasets([ds["train"], ds["validation"], ds["test"]]).map(lambda x: tokenizer(x["summary"], truncation=True), batched=True)
max_completion_len = max([len(x) for x in completions_tokens["input_ids"]])
print(f"Maximo tamagno de completion: {max_completion_len}")


#INICIAR CONVERSACION
print("Escribe 'SALIR' para terminar la conversacion: ")
prompt = ''
while prompt != 'SALIR':
    prompt = input("X: ")

    prompt_tokens = tokenizer_T5(prompt, return_tensors="pt").input_ids.to("cuda")
    outputs = model_T5.generate(prompt_tokens, max_length=100)
    print("Y: " + tokenizer_T5.decode(outputs[0]))

