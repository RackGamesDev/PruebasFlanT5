#PAQUETES NECESARIOS
#!pip install transformers
#!pip install sentencepiece
#!pip install accelerate
#!pip install datasets
#!pip install evaluate
#!pip install rouge_score
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_dataset, concatenate_datasets, Dataset
import torch, os, nltk, evaluate, numpy as np
from nltk.tokenize import sent_tokenize

EVALUAR = True #Si se quiere evaluar el modelo una vez entrenado
NUM_EJ_TRAIN = 1500 #Numero de ejemplos de entrenamiento, validacion y test
NUM_EJ_VAL = 500
NUM_EJ_TEST = 200
VERSION='small' #Version del modelo: small, base, large
REPOSITORY="./flan-t5-" + VERSION + "-fine-tuned-conversational" #UBICACION DEL MODELO ENTRENADO


#DESCARGAR EL DATASET PARA ENTRENAR Y PREPARARLO
ds = load_dataset("daily_dialog", cache_dir="./cachedataset")

def build_prompt(dialog, turn_idx, context_turns=2):
    prompt = []
    start = max(0, turn_idx - context_turns)
    for i in range(start, turn_idx):
        speaker = "User" if i % 2 == 0 else "Bot"
        prompt.append(f"{speaker}: {dialog[i]}")
    return "\n".join(prompt), dialog[turn_idx]

def flatten_dialogs(dataset_split, num_examples, context_turns=2):
    prompts, responses = [], []
    for ex in dataset_split.select(range(num_examples)):
        dialog = ex['dialog']
        for i in range(1, len(dialog)):
            prompt, response = build_prompt(dialog, i, context_turns=context_turns)
            prompts.append(prompt)
            responses.append(response)
    return {"prompt": prompts, "response": responses}
train_data = flatten_dialogs(ds["train"], NUM_EJ_TRAIN)
val_data = flatten_dialogs(ds["validation"], NUM_EJ_VAL)
test_data = flatten_dialogs(ds["test"], NUM_EJ_TEST)
train_dataset = Dataset.from_dict(train_data)
val_dataset = Dataset.from_dict(val_data)
test_dataset = Dataset.from_dict(test_data)
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-" + VERSION)

all_prompts = train_data["prompt"] + val_data["prompt"] + test_data["prompt"]
all_responses = train_data["response"] + val_data["response"] + test_data["response"]
prompts_tokens = [tokenizer(x, truncation=True)["input_ids"] for x in all_prompts]
responses_tokens = [tokenizer(x, truncation=True)["input_ids"] for x in all_responses]
max_token_len = max(len(x) for x in prompts_tokens)
max_response_len = max(len(x) for x in responses_tokens)
print(f"Max prompt length: {max_token_len}")
print(f"Max response length: {max_response_len}")

def padding_tokenizer(batch):
    model_inputs = tokenizer(batch['prompt'], max_length=max_token_len, padding="max_length", truncation=True)
    model_labels = tokenizer(batch['response'], max_length=max_response_len, padding="max_length", truncation=True)
    model_labels["input_ids"] = [[(l if l != tokenizer.pad_token_id else -100) for l in label] for label in model_labels["input_ids"]]
    model_inputs['labels'] = model_labels["input_ids"]
    return model_inputs

train_tokens = train_dataset.map(padding_tokenizer, batched=True, remove_columns=['prompt', 'response'])
val_tokens = val_dataset.map(padding_tokenizer, batched=True, remove_columns=['prompt', 'response'])
test_tokens = test_dataset.map(padding_tokenizer, batched=True, remove_columns=['prompt', 'response'])


#IMPORTAR EL MODELO BASE (flan-t5 de google)
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-" + VERSION, cache_dir="./cachemodel")

#ENTRENAR EL MODELO
nltk.data.path.append('./nltk_data')
nltk.download("punkt", download_dir="./nltk_data")
nltk.download("punkt_tab", download_dir="./nltk_data")

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    preds = np.array(preds)
    preds = np.where((preds < 0) | (preds >= tokenizer.vocab_size), tokenizer.pad_token_id, preds)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    #decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    avg_len = np.mean([len(x.split()) for x in decoded_preds])
    return {"num_samples": len(decoded_preds), "avg_pred_words": avg_len}


label_pad_token_id = -100
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=label_pad_token_id, pad_to_multiple_of=8)

training_args = Seq2SeqTrainingArguments(
    output_dir=REPOSITORY,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    fp16=False,
    learning_rate=5e-5,
    num_train_epochs=4,
    logging_dir=f"{REPOSITORY}/logs",
    logging_strategy="steps",
    logging_steps=500,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    generation_max_length=60,
    generation_num_beams=2,
    #generation_kwargs={"repetition_penalty": 1.2, "num_beams": 4}
)
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_tokens,
    eval_dataset=val_tokens,
    compute_metrics=compute_metrics,
)
tokenizer.save_pretrained(REPOSITORY + "/tokenizer")
trainer.train()


#EVALUAR ENTRENAMIENTO
if EVALUAR:
    tokenizer_FT5_FT = T5Tokenizer.from_pretrained(REPOSITORY + "/tokenizer")
    checkpoints = [d for d in os.listdir(REPOSITORY) if d.startswith("checkpoint")]
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split('-')[-1]))
    last_checkpoint = checkpoints[-1] if checkpoints else None
    model_FT5_FT = T5ForConditionalGeneration.from_pretrained(REPOSITORY + f"/{last_checkpoint}", device_map="auto")
    model_FT5_FT.eval()
    batch_size = 8
    all_predictions = []
    with torch.no_grad():
        for i in range(0, len(test_tokens["input_ids"]), batch_size):
            input_ids_batch = torch.tensor(test_tokens["input_ids"][i:i+batch_size], device='cuda:0')
            outputs = model_FT5_FT.generate(input_ids_batch, max_length=60, do_sample=True, top_p=0.95, temperature=0.7, num_beams=2, repetition_penalty=1.6, no_repeat_ngram_size=3, early_stopping=True)
            all_predictions.extend(outputs)
    decoded_preds = tokenizer_FT5_FT.batch_decode(all_predictions, skip_special_tokens=True)
    decoded_inputs = tokenizer_FT5_FT.batch_decode(test_tokens["input_ids"], skip_special_tokens=True)
    for inp, pred in zip(decoded_inputs[:10], decoded_preds[:10]):
        print(f"User: {inp}\nBot: {pred}\n")



#INICIAR CONVERSACION CON EL MODELO ENTRENADO
print("Escribe 'SALIR' para terminar la conversacion: ")
prompt = ''
while prompt != 'SALIR':
    prompt = input("X: ")
    prompt_tokens = tokenizer_FT5_FT(prompt, return_tensors="pt").input_ids.to("cuda")
    outputs = model_FT5_FT.generate(prompt_tokens, max_length=60, do_sample=True, top_p=0.95, temperature=0.7, num_beams=2, repetition_penalty=1.6, no_repeat_ngram_size=3, early_stopping=True)
    print("Y: " + tokenizer_FT5_FT.decode(outputs[0], skip_special_tokens=True))

model_FT5_FT.save_pretrained('./prod')
tokenizer_FT5_FT.save_pretrained('./prod')
