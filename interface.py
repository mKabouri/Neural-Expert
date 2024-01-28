import tkinter as tk
from tkinter import scrolledtext

class GUI:
    def __init__(self, master, kb):
        self.master = master
        self.kb = kb
        master.title("Neural Network Recommendation Expert System")

        self.create_widgets()

    def create_widgets(self):
        questions = {
            "data_size": "What is the size of your dataset (small, medium, large)?",
            "data_type": "What type of data are you working with (image, text, tabular)?",
            "task_type": "What is your specific task type (classification, regression, etc.)?",
            "inference_speed": "Do you need fast real-time inference (yes, no)?",
            "computational_resource": "Are you limited by computational resources (yes, no)?"
        }

        self.entries = {}
        for i, (key, question) in enumerate(questions.items()):
            label = tk.Label(self.master, text=question)
            label.grid(row=i, column=0)
            entry = tk.Entry(self.master)
            entry.grid(row=i, column=1)
            self.entries[key] = entry

        submit_button = tk.Button(self.master, text="Submit", command=self.process_input)
        submit_button.grid(row=len(questions), column=0)

        clear_button = tk.Button(self.master, text="Clear", command=self.clear_inputs)
        clear_button.grid(row=len(questions), column=1)

        self.output_area = scrolledtext.ScrolledText(self.master, wrap=tk.WORD, height=10, state=tk.DISABLED)
        self.output_area.grid(row=len(questions)+1, column=0, columnspan=2)

    def clear_inputs(self):
        for entry in self.entries.values():
            entry.delete(0, tk.END)
        self.output_area.configure(state=tk.NORMAL)
        self.output_area.delete('1.0', tk.END)
        self.output_area.configure(state=tk.DISABLED)

    def process_input(self):
        for key, entry in self.entries.items():
            response = entry.get().strip().lower()
            if response:
                self.kb.addFact(f"{key}:{response}")
        self.kb.simpleFC()
        self.kb.recommendNN()
        self.display_recommendations()

    def display_recommendations(self):
        self.output_area.configure(state=tk.NORMAL)
        self.output_area.delete('1.0', tk.END)
        recommendations = "\n".join([f"- {fact[3:]}" for fact in self.kb.facts if fact.startswith("NN:")])
        self.output_area.insert(tk.INSERT, recommendations)
        self.output_area.configure(state=tk.DISABLED)
