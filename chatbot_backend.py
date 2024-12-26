from flask import Flask, request, jsonify
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import requests
from flask_cors import CORS  # Pastikan ini sudah di-import

# Inisialisasi Flask
app = Flask(__name__)
CORS(app)  # Menambahkan CORS untuk memungkinkan frontend dari domain lain mengakses API

# Load pre-trained model and tokenizer
model_name = "microsoft/DialoGPT-small"  # Ganti dengan model yang diinginkan
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Function to generate chatbot response
def chatbot_response(input_text, chat_history_ids=None):
    new_input_ids = tokenizer.encode(input_text + tokenizer.eos_token, return_tensors="pt")
    
    # Combine chat history and new input
    if chat_history_ids is not None:
        bot_input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1)
    else:
        bot_input_ids = new_input_ids

    # Generate response
    with torch.no_grad():  # Disable gradients for inference
        chat_history_ids = model.generate(
            bot_input_ids,
            max_length=1000,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.9
        )

    # Decode response
    response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response, chat_history_ids

# Function to access the internet and search for an answer
def search_internet(query):
    api_key = "e2cbd0fc457b6c679857b02b03a59a468e3836f84c4f3ddffd54d61266da5de4"  # Ganti dengan API Key Anda dari SerpAPI
    url = f"https://serpapi.com/search?q={query}&api_key={api_key}"
    
    try:
        response = requests.get(url)
        search_results = response.json()
        if "organic_results" in search_results:
            answer = search_results["organic_results"][0]["snippet"]
            return answer
        else:
            return "Maaf, saya tidak bisa menemukan jawaban di internet."
    except Exception as e:
        return f"Terjadi kesalahan saat mencari di internet: {str(e)}"

# Endpoint untuk chatbot
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_input = data.get("input")
    chat_history_ids = None
    if user_input:
        # Jika input berupa pertanyaan, coba cari jawabannya di internet
        if "?" in user_input:
            internet_answer = search_internet(user_input)
            return jsonify({"response": internet_answer})

        # Dapatkan respons chatbot
        try:
            response, chat_history_ids = chatbot_response(user_input, chat_history_ids)
            return jsonify({"response": response})
        except Exception as e:
            return jsonify({"error": f"Terjadi kesalahan: {str(e)}"}), 500
    return jsonify({"error": "Input tidak valid"}), 400

# Jalankan aplikasi Flask
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)  # pastikan port 5000
