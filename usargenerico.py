from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model = AutoModelForSeq2SeqLM.from_pretrained('./prod')
tokenizer = AutoTokenizer.from_pretrained('./prod')

print('TODO LISTO:')
prompt = ''
while prompt != 'SALIR':
    prompt = input("X: ")
    prompt_tokens = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    outputs = model.generate(prompt_tokens, max_length=60, do_sample=True, top_p=0.95, temperature=0.7, repetition_penalty=1.2, num_beams=2)
    print("Y: " + tokenizer.decode(outputs[0], skip_special_tokens=True))