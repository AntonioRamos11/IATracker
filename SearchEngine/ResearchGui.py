#!/usr/bin/env python3
# filepath: /home/p0wden/Documents/IAResearchAgregator/SearchEngine/ResearchGui.py
import tkinter as tk
from tkinter import ttk, scrolledtext, font
import threading
import time
import random
from ResearchQuestionAnswerer import ResearchQuestionAnswerer

class HackerTheme:
    """Cyberpunk/hacker theme colors and fonts"""
    BG_COLOR = "#0C0C0C"  # Almost black
    TEXT_COLOR = "#00FF41"  # Matrix green
    ACCENT_COLOR = "#0ABDC6"  # Cyberpunk blue
    HIGHLIGHT_COLOR = "#711C91"  # Neon purple
    WARNING_COLOR = "#EA00D9"  # Hot pink
    FONT_FAMILY = "Courier"  # Classic terminal font

class TerminalText(scrolledtext.ScrolledText):
    """Custom scrolled text widget with terminal-like typing effect"""
    
    def __init__(self, master, **kwargs):
        # Set default styling for terminal look
        kwargs.setdefault("bg", HackerTheme.BG_COLOR)
        kwargs.setdefault("fg", HackerTheme.TEXT_COLOR)
        kwargs.setdefault("insertbackground", HackerTheme.ACCENT_COLOR)  # Cursor color
        kwargs.setdefault("font", (HackerTheme.FONT_FAMILY, 10))
        kwargs.setdefault("borderwidth", 1)
        kwargs.setdefault("relief", tk.SUNKEN)
        
        super().__init__(master, **kwargs)
    
    def type_text(self, text, speed=10, callback=None):
        """Display text with a terminal typing effect"""
        self.delete("1.0", tk.END)
        self._type_next_char(text, 0, speed, callback)
    
    def _type_next_char(self, text, index, speed, callback):
        """Recursively type characters with realistic timing"""
        if index < len(text):
            # Type the next character
            self.insert(tk.END, text[index])
            self.see(tk.END)
            
            # Random delay for realistic typing
            delay = int(random.gauss(speed, speed/2))
            delay = max(1, min(delay, speed*3))  # Keep within reasonable bounds
            
            # Schedule next character
            self.after(delay, lambda: self._type_next_char(text, index+1, speed, callback))
        else:
            # Typing finished
            if callback:
                callback()

class ResearchQAGui:
    def __init__(self, root):
        self.root = root
        self.root.title("Neural Research Interface v1.0")
        self.root.geometry("950x750")
        self.root.minsize(800, 600)
        
        # Apply hacker theme
        self.root.configure(bg=HackerTheme.BG_COLOR)
        self.style = ttk.Style()
        self.style.theme_use('alt')  # Use alternative theme as base
        
        # Configure styles
        self.style.configure("TFrame", background=HackerTheme.BG_COLOR)
        self.style.configure("TLabel", 
                             background=HackerTheme.BG_COLOR, 
                             foreground=HackerTheme.TEXT_COLOR,
                             font=(HackerTheme.FONT_FAMILY, 10))
        self.style.configure("TButton", 
                             background=HackerTheme.BG_COLOR, 
                             foreground=HackerTheme.ACCENT_COLOR,
                             font=(HackerTheme.FONT_FAMILY, 10, "bold"))
        self.style.map("TButton",
                      background=[('active', HackerTheme.HIGHLIGHT_COLOR)],
                      foreground=[('active', HackerTheme.BG_COLOR)])
        self.style.configure("TCheckbutton", 
                             background=HackerTheme.BG_COLOR, 
                             foreground=HackerTheme.TEXT_COLOR,
                             font=(HackerTheme.FONT_FAMILY, 10))
        self.style.configure("TLabelframe", 
                             background=HackerTheme.BG_COLOR,
                             foreground=HackerTheme.ACCENT_COLOR,
                             font=(HackerTheme.FONT_FAMILY, 10))
        self.style.configure("TLabelframe.Label", 
                             background=HackerTheme.BG_COLOR,
                             foreground=HackerTheme.ACCENT_COLOR,
                             font=(HackerTheme.FONT_FAMILY, 10, "bold"))
        self.style.configure("Horizontal.TProgressbar", 
                             background=HackerTheme.ACCENT_COLOR)
        self.style.configure("TScale", 
                             background=HackerTheme.BG_COLOR)
        
        # Initialize the QA engine
        self.qa_engine = None
        self.is_engine_ready = False
        
        # Header with ASCII art
        header_frame = ttk.Frame(root)
        header_frame.pack(fill=tk.X, pady=5)
        
        ascii_art = """
  _   _                      _   ____                               _     
 | \\ | | ___ _   _ _ __ __ _| | |  _ \\ ___  ___  ___  __ _ _ __ ___| |__  
 |  \\| |/ _ \\ | | | '__/ _` | | | |_) / _ \\/ __|/ _ \\/ _` | '__/ __| '_ \\ 
 | |\\  |  __/ |_| | | | (_| | | |  _ <  __/\\__ \\  __/ (_| | | | (__| | | |
 |_| \\_|\\___|\\__,_|_|  \\__,_|_| |_| \\_\\___||___/\\___|\\__,_|_|  \\___|_| |_|
                                                                           
        """ 
        header_label = ttk.Label(header_frame, text=ascii_art, font=("Courier", 8))
        header_label.pack()
        
        # Create the main frame
        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Question input area
        question_frame = ttk.LabelFrame(main_frame, text="< RESEARCH QUERY >", padding="10")
        question_frame.pack(fill=tk.X, expand=False, padx=5, pady=5)
        
        self.question_entry = TerminalText(question_frame, height=3, wrap=tk.WORD)
        self.question_entry.pack(fill=tk.X, expand=True, padx=5, pady=5)
        
        # Options frame
        options_frame = ttk.Frame(main_frame, padding="5")
        options_frame.pack(fill=tk.X, expand=False, padx=5, pady=5)
        
        # Quick answer checkbox
        self.quick_var = tk.BooleanVar(value=False)
        quick_check = ttk.Checkbutton(
            options_frame, 
            text="STEALTH MODE [No LLM]", 
            variable=self.quick_var
        )
        quick_check.grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        
        # Top-k papers slider
        ttk.Label(options_frame, text="DATA SOURCES:").grid(row=0, column=1, padx=(20, 5), pady=5)
        self.top_k_var = tk.IntVar(value=5)
        top_k_slider = ttk.Scale(
            options_frame, 
            from_=1, 
            to=10, 
            orient=tk.HORIZONTAL, 
            variable=self.top_k_var,
            length=100
        )
        top_k_slider.grid(row=0, column=2, padx=5, pady=5)
        self.top_k_label = ttk.Label(options_frame, text="5")
        self.top_k_label.grid(row=0, column=3, padx=5, pady=5)
        top_k_slider.configure(command=self.update_top_k_label)
        
        # Temperature slider (only visible for LLM mode)
        ttk.Label(options_frame, text="ENTROPY:").grid(row=0, column=4, padx=(20, 5), pady=5)
        self.temp_var = tk.DoubleVar(value=0.7)
        temp_slider = ttk.Scale(
            options_frame, 
            from_=0.1, 
            to=1.0, 
            orient=tk.HORIZONTAL, 
            variable=self.temp_var,
            length=100
        )
        temp_slider.grid(row=0, column=5, padx=5, pady=5)
        self.temp_label = ttk.Label(options_frame, text="0.7")
        self.temp_label.grid(row=0, column=6, padx=5, pady=5)
        temp_slider.configure(command=self.update_temp_label)
        
        # Submit button
        submit_btn = ttk.Button(main_frame, text="EXECUTE QUERY", command=self.submit_question)
        submit_btn.pack(pady=10)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("SYSTEM IDLE")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM, pady=5)
        
        # Progress bar
        self.progress = ttk.Progressbar(main_frame, orient=tk.HORIZONTAL, mode='indeterminate', style="Horizontal.TProgressbar")
        self.progress.pack(fill=tk.X, side=tk.BOTTOM, pady=5)
        
        # Answer display area
        answer_frame = ttk.LabelFrame(main_frame, text="< NEURAL OUTPUT >", padding="10")
        answer_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.answer_text = TerminalText(answer_frame, wrap=tk.WORD, height=15)
        self.answer_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Initialize welcome message with typing effect
        welcome_msg = ">> NEURAL RESEARCH INTERFACE INITIALIZING...\n\n"
        welcome_msg += ">> ESTABLISHING CONNECTION TO KNOWLEDGE DATABASE...\n"
        welcome_msg += ">> LOADING COGNITIVE ARCHITECTURE...\n"
        welcome_msg += ">> QUANTUM INDEXING ALGORITHM ACTIVATED...\n\n"
        welcome_msg += ">> READY FOR QUERIES. ENTER YOUR RESEARCH QUESTION ABOVE."
        self.answer_text.type_text(welcome_msg, speed=30)
        
        # Initialize engine in background
        self.update_status("INITIALIZING NEURAL CORES...")
        self.progress.start()
        
        threading.Thread(target=self.initialize_engine, daemon=True).start()
        
        # Blinking cursor effect for terminal feel
        self._blink_cursor()
        
    def _blink_cursor(self):
        """Creates a blinking cursor effect in the status bar"""
        current_status = self.status_var.get()
        if current_status.endswith("_"):
            self.status_var.set(current_status[:-1] + " ")
        else:
            self.status_var.set(current_status.rstrip() + "_")
        self.root.after(500, self._blink_cursor)
        
    def initialize_engine(self):
        try:
            self.qa_engine = ResearchQuestionAnswerer()
            
            self.root.after(0, self.update_status, "LOADING PAPER DATABASE...")
            self.qa_engine.load_papers()
            
            self.root.after(0, self.update_status, "BUILDING NEURAL NETWORK CONNECTIONS...")
            self.qa_engine.build_index()
            
            self.is_engine_ready = True
            self.root.after(0, self.update_status, "SYSTEM READY. AWAITING INPUT_")
            self.root.after(0, self.progress.stop)
        except Exception as e:
            self.root.after(0, self.update_status, f"ERROR: {str(e)}")
            self.root.after(0, self.progress.stop)
    
    def update_status(self, message):
        self.status_var.set(message)
    
    def update_top_k_label(self, value):
        self.top_k_label.configure(text=str(int(float(value))))
    
    def update_temp_label(self, value):
        self.temp_label.configure(text=f"{float(value):.1f}")
    
    def submit_question(self):
        if not self.is_engine_ready:
            self.update_status("NEURAL CORES NOT READY. PLEASE WAIT_")
            return
        
        question = self.question_entry.get("1.0", tk.END).strip()
        if not question:
            self.update_status("ERROR: NO QUERY DETECTED_")
            return
        
        self.answer_text.delete("1.0", tk.END)
        self.answer_text.insert("1.0", ">> PROCESSING QUERY...\n>> SEARCHING NEURAL NETWORK FOR RELEVANT DATA...\n")
        self.progress.start()
        
        is_quick = self.quick_var.get()
        top_k = int(self.top_k_var.get())
        temperature = float(self.temp_var.get())
        
        # Update status
        if is_quick:
            self.update_status("RUNNING STEALTH MODE: BYPASSING NEURAL NETS_")
        else:
            self.update_status("ACTIVATING DEEP NEURAL PROCESSING_")
        
        # Process in a separate thread to avoid freezing the UI
        threading.Thread(
            target=self.process_question,
            args=(question, is_quick, top_k, temperature),
            daemon=True
        ).start()
    
    def process_question(self, question, is_quick, top_k, temperature):
        try:
            if is_quick:
                self.root.after(0, self.update_status, "SCANNING DOCUMENT VECTORS_")
                answer = self.qa_engine.quick_answer(question)
            else:
                # Update status for longer process
                self.root.after(0, self.update_status, "INITIALIZING QUANTUM NEURAL NETWORK_")
                answer = self.qa_engine.answer_question(
                    question=question,
                    top_k=top_k,
                    temperature=temperature
                )
            
            # Update UI with the answer using a typing effect
            prefix = ">> ANALYSIS COMPLETE. DISPLAYING RESULTS:\n\n"
            self.root.after(0, lambda: self.answer_text.delete("1.0", tk.END))
            self.root.after(0, lambda: self.answer_text.type_text(
                prefix + answer, 
                speed=5, 
                callback=lambda: self.root.after(0, self.update_status, "OPERATION COMPLETE_")
            ))
        except Exception as e:
            error_msg = f">> ERROR DETECTED: {str(e)}\n>> NEURAL PROCESSING FAILED.\n>> TRY ALTERNATIVE QUERY PARAMETERS."
            self.root.after(0, lambda: self.answer_text.delete("1.0", tk.END))
            self.root.after(0, lambda: self.answer_text.type_text(error_msg, speed=20))
            self.root.after(0, self.update_status, "ERROR: NEURAL PROCESSING FAILURE_")
        finally:
            self.root.after(0, self.progress.stop)
    
    def display_answer(self, answer):
        self.answer_text.delete("1.0", tk.END)
        self.answer_text.insert("1.0", answer)

if __name__ == "__main__":
    root = tk.Tk()
    app = ResearchQAGui(root)
    root.mainloop()