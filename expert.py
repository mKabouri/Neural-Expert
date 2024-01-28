import argparse
import tkinter as tk
from tkinter import scrolledtext

class Rule():
    def __init__(self, strRule):
        self.active = False
        self.pre, self.post = [], []
        [strPre,separator,strPost] = strRule.partition("=>")
        if separator == "": # implication not found
          raise Exception("Missing '=>' in rule") 
        self.pre = [s.strip() for s in strPre.split("&")]
        self.post = [s.strip() for s in strPost.split("&")]
        self.active = True
        
    def __str__(self):
        return str(self.pre) + " => " + str(self.post)

class KB():
    def __init__(self, verb = True):
        self.facts=[]
        self.rules=[]
        self.verb = verb

    def addRule(self,r):
        self.rules.append(r)

    def addFact(self,f):
        print("Adding fact ", f)
        self.facts.append(f)

    def rectractFact(self,f):
        self.facts.remove(f)

    def _propagateRule(self,rule):
        empty = True
        ret = False
        for pre in rule.pre:
            if self.verb: print("evaluating ", pre)
            if pre not in self.facts:
                empty = False
                break
        if empty:
            if self.verb: print("Propagating rule " + str(rule.pre) + " => " + str(rule.post))
            for post in rule.post:
                if post not in self.facts:
                    if self.verb: print("Adding " + post + " as a new fact") 
                    self.addFact(post)
                    ret = True # At least one fact has been added
            rule.active = False
        return ret

    def simpleFC(self):
        "Simple Forward Checking algorithm. No smart data structure used"
        loop = True # True if any fact has been added
        while loop:
            loop = False
            for rule in self.rules:
                if rule.active:
                    loop |= self._propagateRule(rule)

    def _getRulesForFact(self, fact):
        "Returns the list of rules that contains this fact as a post"
        return [rule for rule in self.rules if fact in rule.post]

    def _ask(self, f):
        "Asks for the value of a fact"
        print("Forcing " + str(f) + " as a new Fact")
        self.addFact(f) # By default make it true
        return

    def simpleBC(self, fact):
        "Simple Backward chaining for a fact, returns after any new fact is added after any question"
        print("BC for fact " + str(fact))
        for rule in self._getRulesForFact(fact):
            print(rule)
            for pre in rule.pre:
                if pre not in self.facts:
                    rulespre = self._getRulesForFact(pre)
                    if not rulespre: # no rules conclude on it. This is an askable fact
                        self._ask(pre)
                        return True
                    else:
                        return self.simpleBC(pre)

    def refine_recommendations(self):
        # Example refinements based on combined facts
        if 'data_size:large' in self.facts and 'task_type:classification' in self.facts:
            self.addFact("NN:Consider Using Distributed Training Techniques")

        if 'data_type:image' in self.facts and 'inference_speed:yes' in self.facts:
            self.addFact("NN:Use Models Optimized for Inference Speed, like MobileNet or EfficientNet")

        if 'data_type:text' in self.facts and 'task_type:other' in self.facts:
            self.addFact("NN:Explore Advanced Models like BERT or GPT for Complex Text Tasks")

        if 'computational_resource:no' in self.facts and 'data_size:large' not in self.facts:
            self.addFact("NN:Prioritize Efficient Models to Balance Performance and Resource Usage")
        
        if 'data_type:text' in self.facts and 'task_type:nlp' in self.facts and 'computational_resource:no' in self.facts:
            self.addFact("NN:Consider DistilBERT for Efficient NLP")

        if 'data_type:image' in self.facts and 'task_type:segmentation' in self.facts and 'computational_resource:no' in self.facts:
            self.addFact("NN:Optimize Segmentation Models for Efficiency")

        if 'data_size:very_large' in self.facts and 'task_type:classification' in self.facts:
            self.addFact("NN:Consider Parallel Processing and Model Sharding for Scalability")

        if 'data_type:image' in self.facts and 'task_type:segmentation' in self.facts and 'computational_resource:yes' in self.facts:
            self.addFact("NN:Explore Advanced Vision Transformers for Detailed Image Segmentation")

        if 'data_type:text' in self.facts and 'task_type:sentiment_analysis' in self.facts:
            self.addFact("NN:Use Fine-tuned BERT or DistilBERT Models for Sentiment Analysis")
        if 'data_type:graph' in self.facts and 'task_type:node_classification' in self.facts:
            self.addFact("NN:Use GCN for Node Classification in Graphs")

        if 'data_type:graph' in self.facts and 'task_type:link_prediction' in self.facts:
            self.addFact("NN:Consider GraphSAGE or GAT for Link Prediction")

        if 'data_type:graph' in self.facts and 'task_type:graph_embedding' in self.facts:
            self.addFact("NN:Use VGAE for Graph Embedding Tasks")

        # Refinements for Self-Supervised Learning
        if 'task_type:self_supervised_learning' in self.facts:
            if 'data_type:image' in self.facts:
                self.addFact("NN:Explore Contrastive Learning Methods for Images")
            elif 'data_type:text' in self.facts:
                self.addFact("NN:Consider Masked Language Models for Text")
            elif 'data_type:audio' in self.facts:
                self.addFact("NN:Use CPC for Audio Feature Learning")
            elif 'data_type:video' in self.facts:
                self.addFact("NN:Investigate Video Frame Prediction Models")
            elif 'data_type:graph' in self.facts:
                self.addFact("NN:Use Graph Autoencoders for Node Embedding")

        # Refinements based on Pretext Tasks in Self-Supervised Learning
        if 'pretext_task:image_rotation' in self.facts:
            self.addFact("NN:Use RotNet for Self-Supervised Learning by Image Rotation Prediction")

        if 'pretext_task:word_ordering' in self.facts:
            self.addFact("NN:Use Word Ordering Tasks for Textual Feature Extraction")

        if 'pretext_task:audio_segment_prediction' in self.facts:
            self.addFact("NN:Use Segment Prediction Models for Audio Representation")
        
        # Preprocessing suggestions
        if 'data_type:image' in self.facts:
            self.addFact("Preprocessing:Consider Image Augmentation Techniques like Rotation, Flipping")
            self.addFact("Preprocessing:Normalize Pixel Values")

        if 'data_type:text' in self.facts:
            self.addFact("Preprocessing:Use Tokenization and Embeddings")
            self.addFact("Preprocessing:Consider Removing Stop Words and Applying Stemming")

        if 'data_type:tabular' in self.facts:
            self.addFact("Preprocessing:Normalize or Standardize Features")
            self.addFact("Preprocessing:Handle Missing Values with Imputation")

        if 'data_type:audio' in self.facts:
            self.addFact("Preprocessing:Convert Audio to Spectrograms or MFCCs")
            self.addFact("Preprocessing:Apply Noise Reduction Techniques")

        if 'data_type:video' in self.facts:
            self.addFact("Preprocessing:Frame Extraction and Temporal Sampling")
            self.addFact("Preprocessing:Apply Video Stabilization Techniques")

        if 'data_type:graph' in self.facts:
            self.addFact("Preprocessing:Use Node Embedding Techniques")
            self.addFact("Preprocessing:Normalize Edge Weights")

    def explain_recommendation(self, fact):
        explanation = []
        def backtrack(fact, path):
            rules = self._getRulesForFact(fact)
            for rule in rules:
                if all(pre in self.facts for pre in rule.pre):
                    new_path = f"Fact '{fact}' was concluded based on the rule: {rule}."
                    if new_path not in path:
                        path.append(new_path)
                        for pre in rule.pre:
                            backtrack(pre, path)

        backtrack(fact, explanation)
        explanation.reverse()
        return '\n'.join(explanation)

    def recommendNN(self):
        self.refine_recommendations()
        print("Recommended Neural Network Architectures and Parameters:")
        for fact in self.facts:
            if fact.startswith("NN:"):
                print(f"- {fact[3:]}")
                print("Explanation:")
                print(self.explain_recommendation(fact))
                print("-----------------------------------")

    def ask_user(self):
        print("Please answer the following questions:")
        questions = {
            "data_size": "What is the size of your dataset (small: <10k samples, medium: 10k-1M samples, large: >1M samples, very_large: >10M samples)?\n>> ",
            "data_type": "What type of data are you working with (image, text, tabular, audio, video, graph, multi-modal)? If your type isn't listed, please specify.\n>> ",
            "task_type": "What is your specific task type (classification, regression, object_detection, segmentation, time_series, nlp, translation, sentiment_analysis, speech_recognition, music_generation, style_transfer, generative_tasks, self_supervised_learning, pretext_task)? If your task isn't listed, please specify.\n>> ",
            "pretext_task": "If you selected 'pretext_task' for task type, please specify the task (image_rotation, word_ordering, audio_segment_prediction, etc.). Otherwise, type 'none'.\n>> ",
            "inference_speed": "Do you need fast real-time inference (yes, no)? Examples include real-time video processing or instant response applications.\n>> ",
            "computational_resource": "Are you limited by computational resources (yes, no)? Consider whether you're using a personal device or a high-performance computing environment.\n>> "
        }

        for char, question in questions.items():
            response = input(question + " ").strip().lower()
            if response and response != 'none':
                self.addFact(f"{char}:{response}")

    def define_rules(self):
        self.addRule(Rule("data_size:large & data_type:image => NN:Use Deep CNNs (e.g. Resnets, VGGs...)"))
        self.addRule(Rule("data_size:medium & data_type:image => NN:Use Standard CNN"))
        self.addRule(Rule("data_size:small & data_type:image => NN:Use Lightweight CNN Models like MobileNet"))

        self.addRule(Rule("data_type:text & task_type:classification => NN:Use LSTM or GRU"))
        self.addRule(Rule("data_type:text & task_type:regression => NN:Use Bidirectional LSTM"))
        self.addRule(Rule("data_type:text & task_type:other => NN:Experiment with Transformer Models"))

        self.addRule(Rule("data_type:tabular & task_type:classification => NN:Use Simple Feedforward Network"))
        self.addRule(Rule("data_type:tabular & task_type:regression => NN:Use Feedforward Network with Dropout Layers"))
        self.addRule(Rule("data_type:tabular & task_type:other => NN:Consider Ensemble Methods"))

        self.addRule(Rule("inference_speed:yes & computational_resource:yes => NN:Use EfficientNet for Image Tasks"))
        self.addRule(Rule("inference_speed:no & computational_resource:no => NN:Consider Using Pre-trained Models"))

        self.addRule(Rule("data_size:large & computational_resource:no => NN:Implement Data Parallelism"))
        self.addRule(Rule("data_size:large & computational_resource:yes => NN:Optimize Model Architecture for Memory Efficiency"))

        self.addRule(Rule("data_type:image & task_type:object_detection => NN:Use Faster R-CNN or YOLO"))
        self.addRule(Rule("data_size:small & data_type:tabular & task_type:classification => NN:Use Compact Neural Networks"))
        self.addRule(Rule("task_type:time_series => NN:Use RNN or 1D CNN"))
        
        self.addRule(Rule("data_type:image & task_type:segmentation => NN:Use U-Net or SegNet"))
        self.addRule(Rule("data_type:text & computational_resource:yes => NN:Use Larger Transformer Models like LLama2, GPT or Mistral7B"))
        self.addRule(Rule("inference_speed:no & task_type:nlp => NN:Use BERT or XLNet for NLP Tasks"))

        self.addRule(Rule("data_type:image & task_type:classification => NN:Consider Vision Transformer (ViT) for High-Resolution Images"))
        self.addRule(Rule("data_type:image & task_type:object_detection => NN:Use DETR (Transformer-based) for Object Detection"))

        self.addRule(Rule("data_type:text & task_type:nlp & data_size:large => NN:Use GPT-3 for Generative NLP Tasks"))
        self.addRule(Rule("data_type:text & task_type:classification & computational_resource:yes => NN:Use BERT or RoBERTa for Text Classification"))

        self.addRule(Rule("data_type:image & task_type:segmentation => NN:Consider Transformer Models like SETR for Image Segmentation"))
        self.addRule(Rule("data_type:text & task_type:translation => NN:Use Transformer Models like T5 for Translation Tasks"))
        
        self.addRule(Rule("task_type:object_detection & data_type:image => NN:Use YOLO or SSD for Object Detection"))
        self.addRule(Rule("task_type:segmentation & data_type:image => NN:Consider U-Net for Image Segmentation"))
        self.addRule(Rule("task_type:time_series => NN:Consider LSTM or Temporal Convolutional Networks"))
        self.addRule(Rule("task_type:nlp => NN:Use Transformer-based Models like BERT or GPT-3 for NLP Tasks"))
        self.addRule(Rule("task_type:translation => NN:Use Sequence-to-Sequence Models like Transformer for Language Translation"))
        self.addRule(Rule("task_type:sentiment_analysis & data_type:text => NN:Use Fine-tuned BERT or DistilBERT for Sentiment Analysis"))

        self.addRule(Rule("data_type:text & task_type:sentiment_analysis & computational_resource:yes => NN:Use RoBERTa or XLNet for Advanced Sentiment Analysis"))
        self.addRule(Rule("data_type:image & task_type:object_detection & data_size:large => NN:Use EfficientDet for Scalable and Efficient Object Detection"))
        self.addRule(Rule("data_type:audio & task_type:speech_recognition => NN:Use DeepSpeech or WaveNet for Speech Recognition Tasks"))
        self.addRule(Rule("data_type:tabular & task_type:time_series => NN:Use Temporal Fusion Transformers for Time Series Forecasting"))
        self.addRule(Rule("task_type:generative_tasks & data_type:image => NN:Use Generative Adversarial Networks (GANs) for Image Generation"))
        self.addRule(Rule("data_type:video & task_type:action_recognition => NN:Use 3D CNNs or I3D for Video Action Recognition"))
        self.addRule(Rule("data_type:audio & task_type:music_generation => NN:Use Transformer-based Models like Jukebox for Music Generation"))
        self.addRule(Rule("data_size:very_large & data_type:text & task_type:nlp => NN:Use Distributed Training with Models like GPT-3 or T5"))
        self.addRule(Rule("data_type:multi-modal & task_type:classification => NN:Use Cross-modal Neural Networks for Multi-modal Data"))
        self.addRule(Rule("data_type:image & task_type:style_transfer => NN:Use Neural Style Transfer Techniques for Artistic Image Generation"))
        
        # Rules for Graph Neural Networks
        self.addRule(Rule("data_type:graph & task_type:node_classification => NN:Use Graph Convolutional Networks (GCN) for Node Classification"))
        self.addRule(Rule("data_type:graph & task_type:graph_classification => NN:Use Graph Isomorphism Networks (GIN) for Graph Classification"))
        self.addRule(Rule("data_type:graph & task_type:link_prediction => NN:Use GraphSAGE or GAT for Link Prediction"))
        self.addRule(Rule("data_type:graph & task_type:graph_generation => NN:Use Generative Models like GraphRNN for Graph Generation"))
        self.addRule(Rule("data_type:graph & task_type:recommendation_system => NN:Use PinSage for Graph-based Recommendation Systems"))
        self.addRule(Rule("data_type:graph & task_type:community_detection => NN:Use Spectral Clustering with GNNs for Community Detection"))
        self.addRule(Rule("data_type:graph & task_type:graph_embedding => NN:Use Autoencoder-based Models like VGAE for Graph Embedding"))
        self.addRule(Rule("data_type:graph & task_type:knowledge_graph => NN:Use R-GCN for Knowledge Graph Applications"))
        self.addRule(Rule("data_type:graph & task_type:traffic_prediction => NN:Use Spatio-Temporal Graph Convolutional Networks for Traffic Prediction"))
        self.addRule(Rule("data_type:graph & task_type:molecular_property_prediction => NN:Use Message Passing Neural Networks (MPNN) for Molecular Property Prediction"))
        self.addRule(Rule("data_type:graph & task_type:social_network_analysis => NN:Use Attention-based Graph Neural Networks for Social Network Analysis"))
        self.addRule(Rule("data_type:graph & task_type:3d_shape_analysis => NN:Use Dynamic Graph CNNs for 3D Shape Analysis"))
        
        # Rules for Self-Supervised Learning
        self.addRule(Rule("data_type:image & task_type:self_supervised_learning => NN:Use Contrastive Learning Methods like SimCLR or MoCo for Image Representation Learning"))
        self.addRule(Rule("data_type:text & task_type:self_supervised_learning => NN:Use Masked Language Models like BERT or RoBERTa for Text Representation Learning"))
        self.addRule(Rule("data_type:audio & task_type:self_supervised_learning => NN:Use Contrastive Predictive Coding (CPC) for Audio Feature Learning"))
        self.addRule(Rule("data_type:video & task_type:self_supervised_learning => NN:Use Video Frame Prediction Models for Temporal Feature Learning"))
        self.addRule(Rule("data_type:graph & task_type:self_supervised_learning => NN:Use Graph Autoencoders for Node Embedding Learning"))
        self.addRule(Rule("data_type:multi-modal & task_type:self_supervised_learning => NN:Use Cross-Modal Self-Supervised Learning for Multi-modal Data Fusion"))
        self.addRule(Rule("task_type:self_supervised_learning & computational_resource:yes => NN:Experiment with Large-Scale Pre-training Models like GPT-3 or CLIP"))
        self.addRule(Rule("data_type:image & task_type:pretext_task & pretext_task:image_rotation => NN:Use RotNet for Self-Supervised Learning by Image Rotation Prediction"))
        self.addRule(Rule("data_type:text & task_type:pretext_task & pretext_task:word_ordering => NN:Use Word Ordering Tasks for Textual Feature Extraction in Self-Supervised Learning"))
        self.addRule(Rule("data_type:audio & task_type:pretext_task & pretext_task:audio_segment_prediction => NN:Use Segment Prediction Models for Audio Representation in Self-Supervised Learning"))
        self.addRule(Rule("data_type:tabular & task_type:self_supervised_learning => NN:Use Denoising Autoencoders for Feature Extraction from Tabular Data"))

    def start_interactive_session(self):
        print("Welcome to the Neural Network Architecture Recommender System")
        while True:
            self.ask_user()
            self.simpleFC()
            self.recommendNN()

            if input("Would you like to start a new session? (yes/no): ").lower() != 'yes':
                break
        print("Thank you for using the recommender system!")

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
            "computational_resource": "Yes if using a personal device, no for high computing power!."
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

def launch_tkinter_interface():
    print("Launching tkinter interface...")
    root = tk.Tk()
    kb = KB(verb=False)
    kb.define_rules()
    app = GUI(root, kb)
    root.mainloop()

def launch_terminal_interface():
    print("Launching terminal interface...")
    knowledge_base = KB()
    knowledge_base.define_rules()
    knowledge_base.start_interactive_session()

def main():
    parser = argparse.ArgumentParser(description="Expert System for Neural Network Recommendation")
    parser.add_argument('--interface', choices=['tkinter', 'terminal'], default='terminal', help='Select the interface type (default: terminal)')
    args = parser.parse_args()

    if args.interface == 'tkinter':
        launch_tkinter_interface()
    else:
        launch_terminal_interface()

if __name__ == "__main__":
    main()

