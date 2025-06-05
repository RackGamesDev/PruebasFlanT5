#!/bin/bash
mkdir cachedataset cachemodel flan-t5-small-fine-tuned nltk_data prod
#pip install transformers
#pip install sentencepiece
#pip install accelerate
#pip install datasets
#pip install evaluate
#pip install rouge_score
#python3 iniciar-resumen.py
python3 iniciar-conversacion.py
#python3 usargenerico.py
7z a -v24m modelo/prod.7z prod/*
