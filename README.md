# Pruebas Flan-T5
Un script para descargar Flan-T5-small de Google y entrenarlo con el dataset de mlsum, usando distintas librerías de Python para ello
Deben existir previamente las carpetas: cachedataset, nltk_data, cachemodel, flan-t5-small-fine-tuned (modelo entrenado para resúmenes en español), flan-t5-small-fine-tuned-conversational (modelo entrenado para conversar en inglés)
- Flan-T5-small de Google: https://huggingface.co/google/flan-t5-small
- Dataset mlsum de reciTAL: https://huggingface.co/datasets/reciTAL/mlsum 
- Dataset daily_dialog de li2017dailydialog: https://huggingface.co/datasets/li2017dailydialog/daily_dialog
iniciar-conversacion.py está preparado para una ia conversacional en inglés, y iniciar-resumen.py para otra especializada en resumir artículos en español

## License
This project is licensed under the GNU General Public License v3.0.  
See the [LICENSE](./LICENSE.txt) file for details.
