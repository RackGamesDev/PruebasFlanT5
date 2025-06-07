from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from optimum.onnxruntime import ORTModelForSeq2SeqLM

USAR_ONNX = False

if USAR_ONNX:
    model = ORTModelForSeq2SeqLM.from_pretrained('./prod-onnx', use_cache=False)
    tokenizer = AutoTokenizer.from_pretrained('./prod-onnx')

    print('TODO LISTO:')
    prompt = ''
    while prompt != 'SALIR':
        prompt = input("X: ")
        prompt_tokens = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**prompt_tokens, max_length=60, do_sample=True, top_p=0.95, temperature=0.7, num_beams=2, repetition_penalty=1.7, no_repeat_ngram_size=3, early_stopping=True)
        print("Y: " + tokenizer.decode(outputs[0], skip_special_tokens=True))
else:
    model = AutoModelForSeq2SeqLM.from_pretrained('./prod').to("cuda")
    tokenizer = AutoTokenizer.from_pretrained('./prod')

    print('TODO LISTO:')
    prompt = ''
    while prompt != 'SALIR':
        prompt = input("X: ")
        prompt_tokens = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
        outputs = model.generate(prompt_tokens, max_length=60, do_sample=True, top_p=0.95, temperature=0.7, num_beams=2, repetition_penalty=1.7, no_repeat_ngram_size=3, early_stopping=True)
        print("Y: " + tokenizer.decode(outputs[0], skip_special_tokens=True))
