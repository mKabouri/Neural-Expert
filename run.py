import argparse
import tkinter as tk
from interface import GUI
from expert import KB

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
