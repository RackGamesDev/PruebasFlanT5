# Pruebas Flan-T5
Un script para descargar distintos modelos de IA y entrenarlo con el datasets, usando distintas librerías de Python para ello
Deben existir previamente las carpetas: cachedataset, nltk_data, cachemodel, flan-t5-small-fine-tuned (modelo entrenado para resúmenes en español), flan-t5-small-fine-tuned-conversational (modelo entrenado para conversar en inglés)
- Flan-T5-small de Google: https://huggingface.co/google/flan-t5-small
- Dataset mlsum de reciTAL: https://huggingface.co/datasets/reciTAL/mlsum 
- Dataset daily_dialog de li2017dailydialog: https://huggingface.co/datasets/li2017dailydialog/daily_dialog
- Llama-2-7b-chat-hf de NousResearch y Meta: https://huggingface.co/meta-llama/Llama-2-7b-chat-hf 
## Información variada
- iniciar-conversacion.py está preparado para una ia conversacional en inglés, y iniciar-resumen.py para otra especializada en resumir artículos en español
- El modelo se puede descomprimir (llevarlo a la carpeta prod o prod-onnx desde modelo) y probar con usargenerico.py
- Ahora mismo, en la carpeta modelo están compresos los modelos en formato normal de Python y en formato ONNX, preparados para descomprimir en sus respectivas carpetas prod
## Generación de imágenes
En la carpeta "imagenes" hay distintos scripts ajenos de python que generan o editan imagenes usando los modelos citados mas abajo
-StableDiffussion-2-1-base de stabilityai: https://huggingface.co/stabilityai/stable-diffusion-2-1-base 
-Agradecimientos a CompVis por las siguientes imágenes: (https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png) (https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png)
-Agradecimientos a TheLastBen por el cuaderno de Google Colab .ipnyb en la carpeta imagenes

## License
This project is licensed under the GNU General Public License v3.0.  
See the [LICENSE](./LICENSE.txt) file for details.
