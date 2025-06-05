#PAQUETES NECESARIOS
#!pip install transformers
#!pip install sentencepiece
#!pip install accelerate
#!pip install datasets
#!pip install evaluate
#!pip install rouge_score
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_dataset, concatenate_datasets
import torch, nltk, evaluate, numpy as np
from nltk.tokenize import sent_tokenize

EVALUAR = True #Si se quiere evaluar el modelo una vez entrenado
NUM_EJ_TRAIN = 1500 #Numero de ejemplos de entrenamiento, validacion y test
NUM_EJ_VAL = 500
NUM_EJ_TEST = 200
VERSION='small' #Version del modelo: small, base, large
FINETUNING = True #Si se entrenara usando finetuning (o simplemente se usara el base)
REPOSITORY="./flan-t5-" + VERSION + "-fine-tuned" #UBICACION DEL MODELO ENTRENADO


#DESCARGAR EL DATASET PARA ENTRENAR Y PREPARARLO
#El dataset tiene muchos articulos y resumenes asociados a estos (y mas informacion), hay que reducirlo, prepararlo, tokenizarlo (convertir cada palabra en un numero para el diccionario), aplicar padding (que todas las secuencias de numeros midan lo mismo)
ds = load_dataset("mlsum", 'es', cache_dir="./cachedataset")

ds['train'] = ds['train'].select(range(NUM_EJ_TRAIN))
ds['validation'] = ds['validation'].select(range(NUM_EJ_VAL))
ds['test'] = ds['test'].select(range(NUM_EJ_TEST))

def parse_dataset(ejemplo):
    return {"prompt": f"Resumir el siguiente articulo: {ejemplo['text']}"}
ds["train"] = ds["train"].map(parse_dataset)
ds["validation"] = ds["validation"].map(parse_dataset)
ds["test"] = ds["test"].map(parse_dataset)
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-" + VERSION)

prompts_tokens = concatenate_datasets([ds["train"], ds["validation"], ds["test"]]).map(lambda x: tokenizer(x["prompt"], truncation=True), batched=True)
max_token_len = max([len(x) for x in prompts_tokens["input_ids"]])
print(f"Maximo tamagno de prompt: {max_token_len}")
completions_tokens = concatenate_datasets([ds["train"], ds["validation"], ds["test"]]).map(lambda x: tokenizer(x["summary"], truncation=True), batched=True)
max_completion_len = max([len(x) for x in completions_tokens["input_ids"]])
print(f"Maximo tamagno de completion: {max_completion_len}")

def padding_tokenizer(datos):
  model_inputs = tokenizer(datos['prompt'], max_length=max_token_len, padding="max_length", truncation=True)
  model_labels = tokenizer(datos['summary'], max_length=max_completion_len, padding="max_length", truncation=True)
  model_labels["input_ids"] = [[(l if l != tokenizer.pad_token_id else -100) for l in label] for label in model_labels["input_ids"]]
  model_inputs['labels'] = model_labels["input_ids"]
  return model_inputs
ds_tokens = ds.map(padding_tokenizer, batched=True, remove_columns=['text', 'summary', 'topic', 'url', 'title', 'date', 'prompt'])


#IMPORTAR EL MODELO BASE (flan-t5 de google)
if FINETUNING:
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-" + VERSION, cache_dir="./cachemodel")
else:
    tokenizer_T5 = T5Tokenizer.from_pretrained("google/flan-t5-" + VERSION, cache_dir="./cachemodel")
    model_T5 = T5ForConditionalGeneration.from_pretrained("google/flan-t5-" + VERSION, device_map="auto", cache_dir="./cachemodel")

#ENTRENAR EL MODELO
if FINETUNING:
    nltk.data.path.append('./nltk_data')
    nltk.download("punkt", download_dir="./nltk_data")
    nltk.download("punkt_tab", download_dir="./nltk_data")
    metric = evaluate.load("rouge")

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]
        preds = ["\n".join(sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(sent_tokenize(label)) for label in labels]
        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        result = {k: round(v * 100, 4) for k, v in result.items()}
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        return result

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
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=ds_tokens["train"],
        eval_dataset=ds_tokens["validation"],
        compute_metrics=compute_metrics,
    )
    tokenizer.save_pretrained(REPOSITORY + "/tokenizer")
    trainer.train()#.input_ids.to("cuda")


#EVALUAR ENTRENAMIENTO
if FINETUNING and EVALUAR:
    tokenizer_FT5_FT = T5Tokenizer.from_pretrained(REPOSITORY + "/tokenizer")
    model_FT5_FT = T5ForConditionalGeneration.from_pretrained(REPOSITORY + "/checkpoint-752", device_map="auto")

    model_FT5_FT.eval()
    batch_size = 8
    all_predictions = []
    with torch.no_grad():
        for i in range(0, len(ds_tokens["test"]["input_ids"]), batch_size):
            input_ids_batch = torch.tensor(ds_tokens["test"]["input_ids"][i:i+batch_size], device='cuda:0')
            outputs = model_FT5_FT.generate(input_ids_batch)
            all_predictions.extend(outputs)
    labels = np.array(ds_tokens['test']['labels'])
    completions = np.array([pred.cpu().numpy() for pred in all_predictions])
    metrics = compute_metrics((completions, labels))
    print(metrics)



#INICIAR CONVERSACION (IMPORTAR MODELO ENTRENADO EN CASO DE HABER USADO FINETUNING)
print("Escribe 'SALIR' para terminar la conversacion: ")
prompt = ''
while prompt != 'SALIR':
    prompt = input("X: ")

    if FINETUNING:
        
        prompt_tokens = tokenizer_FT5_FT(prompt, return_tensors="pt").input_ids.to("cuda")
        outputs = model_FT5_FT.generate(prompt_tokens, max_length=300)
        print("Y: " + tokenizer_FT5_FT.decode(outputs[0]))
    else:
        prompt_tokens = tokenizer_T5(prompt, return_tensors="pt").input_ids.to("cuda")
        outputs = model_T5.generate(prompt_tokens, max_length=100)
        print("Y: " + tokenizer_T5.decode(outputs[0]))

if FINETUNING:
    model_FT5_FT.save_pretrained('./prod')
    tokenizer_FT5_FT.save_pretrained('./prod')
else:
    model_T5.save_pretrained('./prod')
    tokenizer_T5.save_pretrained('./prod')
