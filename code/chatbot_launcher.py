import customtkinter as ctk
import subprocess

class ModelSelectionApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.geometry("400x350")
        self.title("Seleccionar Modelo y Entrenar")

        ctk.CTkLabel(self, text="Seleccione un modelo para iniciar el chatbot", font=("Helvetica", 16)).pack(pady=20)

        ctk.CTkButton(self, text="Modelo Bayesiano", command=self.launch_bayes_chatbot, width=20, height=2).pack(pady=10)
        ctk.CTkButton(self, text="Modelo de Red Neuronal", command=self.launch_neural_chatbot, width=20, height=2).pack(pady=10)

        ctk.CTkLabel(self, text="Entrenar Modelos:", font=("Helvetica", 16)).pack(pady=20)
        ctk.CTkButton(self, text="Entrenar Modelo Bayesiano", command=self.train_bayes_model, width=25, height=2).pack(pady=10)
        ctk.CTkButton(self, text="Entrenar Modelo de Red Neuronal", command=self.train_neural_model, width=25, height=2).pack(pady=10)

    def launch_bayes_chatbot(self):
        self.destroy()
        import chatbot_app_bayes

    def launch_neural_chatbot(self):
        self.destroy()
        import chatbot_app_neural

    def train_bayes_model(self):
        subprocess.Popen(["python", "train_bayes.py"])

    def train_neural_model(self):
        subprocess.Popen(["python", "train_neural.py"])

if __name__ == "__main__":
    app = ModelSelectionApp()
    app.mainloop()

