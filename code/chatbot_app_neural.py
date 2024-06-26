import customtkinter as ctk
from PIL import Image, ImageTk
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import random  # Importar módulo random para seleccionar respuestas al azar

class MyFrame(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        
        self.arrow = self.load_image("Arrow.png", (30, 30))
        self.logo = self.load_image("Logo.png", (50, 50))

        self.arrow_label = ctk.CTkLabel(self, image=self.arrow, text="") if self.arrow else ctk.CTkLabel(self, text="Image not found")
        self.arrow_label.grid(row=0, column=0, padx=10, pady=10)

        self.logo_label = ctk.CTkLabel(self, image=self.logo, text="") if self.logo else ctk.CTkLabel(self, text="Image not found")
        self.logo_label.grid(row=0, column=1, padx=0, pady=10)

        self.label1 = ctk.CTkLabel(self, text="Tec. Inc")
        self.label1.grid(row=0, column=2, padx=10, pady=0)

    def load_image(self, path, size):
        try:
            img = Image.open(path)
            img = img.resize(size)
            return ImageTk.PhotoImage(img)
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            return None

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.geometry("370x600")
        self.grid_rowconfigure(0, weight=0)
        self.grid_columnconfigure(0, weight=1)

        # Frame de imágenes
        self.my_frame = MyFrame(master=self, height=150, corner_radius=0)
        self.my_frame.grid(row=0, column=0, padx=0, pady=0, sticky="ew")
        self.my_frame.grid_propagate(True)

        # Frame para mostrar la conversación
        self.conversation_frame = ctk.CTkFrame(self, corner_radius=10)
        self.conversation_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        self.conversation_frame.grid_rowconfigure(0, weight=1)
        self.conversation_frame.grid_columnconfigure(0, weight=1)

        self.conversation_text = ctk.CTkTextbox(self.conversation_frame, width=320, height=400)
        self.conversation_text.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # Entry
        self.entry = ctk.CTkEntry(master=self, placeholder_text="Mensaje", width=320, height=40, corner_radius=20)
        self.entry.place(relx=0.5, rely=0.92, anchor="center")

        # Evento entry
        self.entry.bind("<Return>", self.handle_enter)

        self.grid_rowconfigure(2, weight=1)

        # Cargar modelos y preprocesadores
        self.load_models()

        # Banco de respuestas predefinidas
        self.responses = {
            "Technical issue": [
                "It seems you have a technical issue. Can you provide more details?",
                "I'm sorry, it looks like there's a technical problem. Could you describe it in more detail?",
                "I understand you have a technical problem. Let's try to solve it."
            ],
            "Refund request": [
            "Had a problem with payment? let me help you",
                "Sorry to hear you need a refund. Can you give me more details?",
                "I can help with your refund request. Can you provide more information?"
            ],
            "Cancellation request": [
                "It looks like you want to cancel something. Can you tell me more about it?",
                "Do you need help canceling? Let me know what you need to cancel.",
                "I understand you want to cancel something. Could you give me more details?"
            ],
            "Product inquiry": [
                "Do you have a question about a product? I'm here to help.",
                "It seems you have a product inquiry. How can I assist you?",
                "Tell me more about your product query, and I'll be happy to help."
            ],
            "Billing inquiry": [
                "Do you have a billing question? Let me know how I can help.",
                "It looks like there's a billing inquiry. Could you give me more details?",
                "I'm here to help with your billing inquiry. Can you provide more information?"
            ]
        }

    def load_models(self):
        # Cargar modelo de red neuronal
        self.nn_model = tf.keras.models.load_model('nn_model.h5')

        # Cargar tokenizer y label_encoder
        with open('tokenizer.pkl', 'rb') as f:
            self.tokenizer = pickle.load(f)

        with open('label_encoder.pkl', 'rb') as f:
            self.label_encoder = pickle.load(f)

        # Cargar la longitud máxima utilizada durante el entrenamiento
        with open('max_length.pkl', 'rb') as f:
            self.max_length = pickle.load(f)

    def handle_enter(self, event):
        message = self.entry.get()
        if message:
            self.display_message("You", message)
            self.entry.delete(0, "end")
            self.bot_response(message)

    def display_message(self, sender, message):
        self.conversation_text.insert("end", f"{sender}: {message}\n")
        self.conversation_text.yview_moveto(1)

    def bot_response(self, user_message):
        response = self.nn_response(user_message)
        self.display_message("Chatbot", response)

    def nn_response(self, user_message):
        # Preprocesar el mensaje del usuario
        sequence = self.tokenizer.texts_to_sequences([user_message])
        sequence_pad = pad_sequences(sequence, maxlen=self.max_length, padding='post')
        prediction = self.nn_model.predict(sequence_pad)
        category_encoded = np.argmax(prediction)
        category = self.label_encoder.inverse_transform([category_encoded])[0]

        # Seleccionar una respuesta aleatoria de la categoría predicha
        if category in self.responses:
            return random.choice(self.responses[category])
        else:
            return "I'm not sure how to help with that."

app = App()
app.mainloop()
