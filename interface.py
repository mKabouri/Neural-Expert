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
            "data_size": "Dataset size:",
            "data_type": "Data type:",
            "task_type": "Task type:",
            "pretext_task": "Pretext task (if applicable):",
            "inference_speed": "Need for fast real-time inference:",
            "computational_resource": "Computational resource limitations:"
        }

        explanations = {
            "data_size": "Small: <10k samples, Medium: 10k-1M samples, Large: >1M samples, Very Large: >10M samples.",
            "data_type": "Types: image, text, tabular, audio, video, graph, multi-modal. Specify if not listed.",
            "task_type": "Examples: classification, regression, object_detection, etc. Specify if not listed.",
            "pretext_task": "Specify the pretext task for self-supervised learning, or type 'none'.",
            "inference_speed": "Yes for applications like real-time video processing. No otherwise.",
            "computational_resource": "Yes if using a personal device, no for high-performance computing."
        }

        self.entries = {}
        for i, (key, question) in enumerate(questions.items()):
            label = tk.Label(self.master, text=question)
            label.grid(row=2*i, column=0, sticky='w')
            explanation = tk.Label(self.master, text=explanations[key], fg='gray')
            explanation.grid(row=2*i+1, column=0, sticky='w')
            entry = tk.Entry(self.master)
            entry.grid(row=2*i, column=1, sticky='w')
            self.entries[key] = entry

        row_offset = 2 * len(questions)
        submit_button = tk.Button(self.master, text="Submit", command=self.process_input)
        submit_button.grid(row=row_offset, column=0)

        clear_button = tk.Button(self.master, text="Clear", command=self.clear_inputs)
        clear_button.grid(row=row_offset, column=1)

        self.output_area = scrolledtext.ScrolledText(self.master, wrap=tk.WORD, height=10, state=tk.DISABLED)
        self.output_area.grid(row=row_offset+1, column=0, columnspan=2, sticky='we')

    def clear_inputs(self):
        for entry in self.entries.values():
            entry.delete(0, tk.END)
        self.output_area.configure(state=tk.NORMAL)
        self.output_area.delete('1.0', tk.END)
        self.output_area.configure(state=tk.DISABLED)

    def process_input(self):
        self.kb.facts.clear()  # Clear previous facts
        for key, entry in self.entries.items():
            response = entry.get().strip().lower()
            if response and response != 'none':
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
