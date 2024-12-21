from flask import Flask, render_template, request
import torch
from torch import nn
import numpy as np
from model import EncoderRNN, AttnDecoderRNN, Lang, generate_text  # Assurez-vous d'importer vos modèles correctement
import pickle

app = Flask(__name__, template_folder='templates')

# Charger les modèles PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hidden_size = 128


# Load vocabularies
with open('input_lang.pkl', 'rb') as f:
    input_lang_vocab = pickle.load(f)
with open('output_lang.pkl', 'rb') as f:
    output_lang_vocab = pickle.load(f)

# Charger les langages et les modèles sauvegardés
input_lang_vocab_size = 4748  # Changez cette valeur selon votre vocabulaire
output_lang_vocab_size = 3079
encoder = EncoderRNN(input_lang_vocab_size, hidden_size).to(device)
decoder = AttnDecoderRNN(hidden_size, output_lang_vocab_size).to(device)

encoder.load_state_dict(torch.load("encoder.pth", map_location=torch.device('cpu')))
decoder.load_state_dict(torch.load("decoder.pth", map_location=torch.device('cpu')))

encoder.eval()
decoder.eval()

@app.route('/')
def home():
    return render_template('translation.html')

@app.route('/generate', methods=['GET', 'POST'])
def generate():
    if request.method == "POST":
        text = request.form.get("Text")

        generated_text = generate_text(text, encoder, decoder, input_lang_vocab, output_lang_vocab, device)
        generated_text = " ".join(item for item in generated_text if item not in ['<SOS>', '<EOS>'])
    else:
        generated_text = ""
    
    return render_template("translation.html", output=generated_text)

if __name__ == "__main__":
    app.run(debug=True)
