import customtkinter as ctk
from PIL import Image, ImageTk
import pickle
import nltk
from nltk.classify import NaiveBayesClassifier
import random
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

        self.responses = {
    "Technical issue": [
        "It seems you have a technical issue. Could you give me more details?",
        "I'm sorry, it seems there's a technical problem. Could you describe it in more detail?",
        "I understand you're experiencing a technical issue. Let's solve it."
    ],
    "Refund request": [
        "Had a problem with payment? let me help you",
        "I'm sorry you have to request a refund. Can you provide more details?",
        "I can assist you with your refund request. Can you give me more information?"
    ],
    "Cancellation request": [
        "It seems you want to cancel something. Can you tell me more about it?",
        "Do you need help with cancellation? Tell me what you need to cancel.",
        "I understand you wish to cancel something. Could you provide more details?"
    ],
    "Product inquiry": [
        "Do you have any questions about a product? I'm here to help.",
        "It seems you have an inquiry about a product. How can I assist you?",
        "Tell me more about your product inquiry and I'll be happy to help."
    ],
    "Billing inquiry": [
        "Do you have a question about billing? Let me know how I can assist you.",
        "It seems there's a billing inquiry. Could you provide more details?",
        "I'm here to assist you with your billing inquiry. Can you provide more information?"
    ]
}


        # Cargar modelos y preprocesadores
        self.load_models()

    def load_models(self):
        try:
            with open('bayes_classifier.pkl', 'rb') as f:
                self.bayes_classifier = pickle.load(f)
                print("Modelo Naive Bayes cargado exitosamente.")
        except FileNotFoundError:
            print("Error: Archivo bayes_classifier.pkl no encontrado.")

        try:
            with open('unique_words.pkl', 'rb') as f:
                self.unique_words = pickle.load(f)
                print("Palabras únicas cargadas exitosamente.")
        except FileNotFoundError:
            print("Error: Archivo unique_words.pkl no encontrado.")

    def handle_enter(self, event):
        message = self.entry.get()
        if message:
            self.display_message("Tú", message)
            self.entry.delete(0, "end")
            self.bot_response(message)

    def display_message(self, sender, message):
        self.conversation_text.insert("end", f"{sender}: {message}\n")
        self.conversation_text.yview_moveto(1)

    def bot_response(self, user_message):
        response = self.bayes_response(user_message)
        self.display_message("Chatbot", response)

    def bayes_response(self, user_message):
        try:
            words = user_message.split()
            features = {f'contains({word})': (word in words) for word in self.unique_words}
            prediction = self.bayes_classifier.classify(features)
            print(f"Mensaje: {user_message} -> Predicción: {prediction}")

            if prediction in self.responses:
                # Selecciona una frase aleatoria de la lista correspondiente a la etiqueta predicha
                response = random.choice(self.responses[prediction])
            else:
                response = "Lo siento, no puedo procesar tu solicitud en este momento."
            
            return response
        except AttributeError:
            print("Error en la clasificación del modelo Naive Bayes.")
            return "No puedo entender eso en este momento."

app = App()
app.mainloop()
