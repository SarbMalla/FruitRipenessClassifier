import os
import random
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class FruitRipenessGame:
    def __init__(self, root):
        self.root = root
        self.root.title("Fruit Ripeness Classification Game")
        self.root.geometry("800x700")
        self.root.resizable(True, True)

        # Add keyboard bindings
        self.root.bind('r', lambda e: self.classify("ripe"))
        self.root.bind('u', lambda e: self.classify("not_ripe"))

        self.images_folder = "images"
        self.all_files = []
        self.training_files = []
        self.testing_files = []
        self.training_data = []
        self.training_labels = []
        self.testing_labels = []
        self.predictions = []
        self.current_image_index = 0
        self.is_training = True
        self.training_mode = "continuous"  # "continuous" or "single"
        self.classifier = RandomForestClassifier(n_estimators=100)
        self.image_size = (300, 300)
        self.thumbnail_size = (100, 100)
        self.model_trained = False
        self.training_iterations = 0

        self.setup_ui()
        self.load_images()
        self.show_current_image()

    def setup_ui(self):
        # Create custom styles
        style = ttk.Style()
        style.configure("Action.TButton", 
                       font=("Arial", 12, "bold"),
                       padding=10)

        self.main_frame = ttk.Frame(self.root, padding="20")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.title_label = ttk.Label(self.main_frame, text="Training Phase: Classify Images",
                                     font=("Arial", 16, "bold"))
        self.title_label.pack(pady=10)

        self.status_frame = ttk.Frame(self.main_frame)
        self.status_frame.pack(fill=tk.X, pady=5)

        self.status_label = ttk.Label(self.status_frame, text="Image 1/0", font=("Arial", 12))
        self.status_label.pack(side=tk.LEFT, padx=5)

        self.model_status = ttk.Label(self.status_frame, text="Model: Not Trained", font=("Arial", 12))
        self.model_status.pack(side=tk.RIGHT, padx=5)

        self.image_frame = ttk.Frame(self.main_frame, borderwidth=2, relief="solid")
        self.image_frame.pack(pady=10)

        self.image_label = ttk.Label(self.image_frame)
        self.image_label.pack(padx=10, pady=10)

        self.buttons_frame = ttk.Frame(self.main_frame)
        self.buttons_frame.pack(pady=15)

        self.ripe_button = ttk.Button(self.buttons_frame, text="Ripe", command=lambda: self.classify("ripe"))
        self.ripe_button.grid(row=0, column=0, padx=10)

        self.not_ripe_button = ttk.Button(self.buttons_frame, text="Not Ripe",
                                          command=lambda: self.classify("not_ripe"))
        self.not_ripe_button.grid(row=0, column=1, padx=10)

        self.train_model_button = ttk.Button(self.buttons_frame, text="Train Model", command=self.train_model)
        self.train_model_button.grid(row=0, column=2, padx=10)

        self.test_button = ttk.Button(self.buttons_frame, text="Test Model", command=self.start_testing,
                                      state=tk.DISABLED)
        self.test_button.grid(row=0, column=3, padx=10)

        self.info_frame = ttk.Frame(self.main_frame)
        self.info_frame.pack(pady=10, fill=tk.X)

        self.info_text = tk.Text(self.info_frame, height=4, width=70, wrap=tk.WORD)
        self.info_text.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.info_text.insert(tk.END, "Classify images as 'Ripe' or 'Not Ripe'. The model uses color and "
                                      "texture features from your classifications to learn patterns. "
                                      "Continue training as many images as you like before testing.")
        self.info_text.config(state=tk.DISABLED)

        scrollbar = ttk.Scrollbar(self.info_frame, command=self.info_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.info_text.config(yscrollcommand=scrollbar.set)

        # Add keyboard shortcut info
        keyboard_info = ttk.Label(self.main_frame, 
                                text="Keyboard shortcuts: 'r' for Ripe, 'u' for Unripe",
                                font=("Arial", 10))
        keyboard_info.pack(pady=5)

        # Add progress indicator
        self.progress_frame = ttk.Frame(self.main_frame)
        self.progress_frame.pack(fill=tk.X, pady=5)
        
        self.progress_label = ttk.Label(
            self.progress_frame, 
            text=f"Training Progress:  images classified", 
            font=("Arial", 10)
        )
        self.progress_label.pack(side=tk.LEFT)

    def load_images(self):
        if not os.path.exists(self.images_folder):
            os.makedirs(self.images_folder)
            messagebox.showerror("Images folder not found", "Please add images to the 'images' folder and restart.")
            return

        valid_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.jfif')
        self.all_files = [f for f in os.listdir(self.images_folder) if f.lower().endswith(valid_extensions)]
        if not self.all_files:
            messagebox.showerror("No images found", "No image files found in the images folder.")
            return

        # Shuffle all files
        random.shuffle(self.all_files)

        # Use 80% for training initially
        split_idx = int(len(self.all_files) * 0.8)
        self.training_files = self.all_files[:split_idx]
        self.testing_files = self.all_files[split_idx:]

        self.testing_labels = [None] * len(self.testing_files)
        self.predictions = [None] * len(self.testing_files)

        self.status_label.config(text=f"Training: Image 1/{len(self.training_files)}")

    def update_training_set(self):
        """Shuffle the files and recreate training set"""
        # Re-shuffle all files for continuous training
        random.shuffle(self.all_files)
        split_idx = int(len(self.all_files) * 0.8)
        self.training_files = self.all_files[:split_idx]
        self.testing_files = self.all_files[split_idx:]
        self.testing_labels = [None] * len(self.testing_files)
        self.predictions = [None] * len(self.testing_files)
        self.current_image_index = 0

    def show_current_image(self):
        if self.is_training and self.current_image_index < len(self.training_files):
            current_file = self.training_files[self.current_image_index]
            self.show_image(current_file, self.image_label, self.image_size)
            self.status_label.config(text=f"Training: Image {self.current_image_index + 1}/{len(self.training_files)}")
        elif not self.is_training and self.current_image_index < len(self.testing_files):
            current_file = self.testing_files[self.current_image_index]
            self.show_image(current_file, self.image_label, self.image_size)
            self.status_label.config(text=f"Testing: Image {self.current_image_index + 1}/{len(self.testing_files)}")

    def show_image(self, filename, label_widget, size):
        try:
            image_path = os.path.join(self.images_folder, filename)
            img = Image.open(image_path)
            img = img.resize(size, Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            label_widget.config(image=photo)
            label_widget.image = photo
            return True
        except Exception as e:
            print(f"Error loading image {filename}: {e}")
            label_widget.config(image='', text="[Image Error]")
            return False

    def extract_features(self, filename):
        """Extract color and texture features from an image"""
        try:
            image_path = os.path.join(self.images_folder, filename)
            img = Image.open(image_path)
            img = img.resize((50, 50))  # Resize for consistency
            img_array = np.array(img)

            features = []

            # Color features - average color per channel
            if len(img_array.shape) == 3 and img_array.shape[2] >= 3:
                # RGB image
                for channel in range(3):
                    channel_avg = np.mean(img_array[:, :, channel])
                    channel_std = np.std(img_array[:, :, channel])
                    features.extend([channel_avg, channel_std])

                # Calculate ratio of red to green (can indicate ripeness)
                if np.mean(img_array[:, :, 1]) > 0:  # Avoid division by zero
                    r_g_ratio = np.mean(img_array[:, :, 0]) / np.mean(img_array[:, :, 1])
                    features.append(r_g_ratio)
                else:
                    features.append(0)
            else:
                # Grayscale image
                features.extend([np.mean(img_array), np.std(img_array), 0])

            # Texture features - simple edge detection
            if len(img_array.shape) == 3:
                gray = np.mean(img_array, axis=2).astype(np.uint8)
            else:
                gray = img_array.astype(np.uint8)

            # Horizontal and vertical gradients
            h_grad = np.abs(gray[:, 1:] - gray[:, :-1]).mean()
            v_grad = np.abs(gray[1:, :] - gray[:-1, :]).mean()
            features.extend([h_grad, v_grad])

            # Normalize features
            features = np.array(features) / 255.0

            # Ensure fixed length
            padded = np.zeros(12)  # Ensure consistent size
            padded[:min(len(features), 12)] = features[:min(len(features), 12)]

            return padded
        except Exception as e:
            print(f"Error extracting features from {filename}: {e}")
            # Return a random feature vector if image can't be processed
            return np.random.rand(12)

    def classify(self, label):
        if self.is_training:
            current_file = self.training_files[self.current_image_index]
            features = self.extract_features(current_file)
            self.training_data.append(features)
            self.training_labels.append(label)
            self.current_image_index += 1

            # Update progress indicator
            self.progress_label.config(
                text=f"Training Progress: {len(self.training_labels)} images classified"
            )

            # If reached end of training set
            if self.current_image_index >= len(self.training_files):
                # In continuous mode, reshuffle and continue
                if self.training_mode == "continuous":
                    self.training_iterations += 1
                    self.update_training_set()
                    self.show_current_image()
                    self.info_text.config(state=tk.NORMAL)
                    self.info_text.delete(1.0, tk.END)
                    self.info_text.insert(tk.END, 
                        f"Training iteration {self.training_iterations} completed.\n"
                        f"Classified {len(self.training_labels)} images so far.\n"
                        f"{'You can now train the model!' if len(self.training_labels) >= 15 else f'Please classify {15 - len(self.training_labels)} more images.'}")
                    self.info_text.config(state=tk.DISABLED)
            else:
                self.show_current_image()

            # Enable train button if enough samples
            if len(self.training_labels) >= 15 and not self.model_trained:
                self.train_model_button.config(state=tk.NORMAL)
        else:
            # Testing phase
            self.testing_labels[self.current_image_index] = label
            predicted = self.predictions[self.current_image_index]

            # Show feedback
            self.info_text.config(state=tk.NORMAL)
            self.info_text.delete(1.0, tk.END)
            if label == predicted:
                self.info_text.insert(tk.END,
                                      f"Your classification matches the model's prediction! Both classified as '{label}'.")
            else:
                self.info_text.insert(tk.END,
                                      f"Difference detected. You classified as '{label}', but the model predicted '{predicted}'.")
            self.info_text.config(state=tk.DISABLED)

            self.current_image_index += 1

            if self.current_image_index >= len(self.testing_files):
                self.show_results()
            else:
                self.show_current_image()

    def train_model(self):
        if len(self.training_labels) < 15:
            messagebox.showwarning("Insufficient Training Data", 
                                 "Please classify at least 15 images before training.\n"
                                 f"Current: {len(self.training_labels)} images\n"
                                 f"Needed: {15 - len(self.training_labels)} more images")
            return

        if len(set(self.training_labels)) < 2:
            messagebox.showwarning("Insufficient Variety", 
                                 "Please classify some fruits as both 'Ripe' and 'Not Ripe'.\n"
                                 "You need examples of both categories.")
            return

        # Update info
        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(tk.END,
                            "Training the model with your classifications... This uses machine learning to identify patterns in color and texture features.")
        self.info_text.config(state=tk.DISABLED)
        self.root.update()

        # Train the model
        self.classifier.fit(np.array(self.training_data), self.training_labels)
        self.model_trained = True

        # Update status
        self.model_status.config(text=f"Model: Trained ({len(self.training_labels)} images)")
        self.test_button.config(state=tk.NORMAL)

        # Show training metrics
        ripe_count = self.training_labels.count("ripe")
        not_ripe_count = self.training_labels.count("not_ripe")

        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(tk.END, 
                            f"Model trained successfully with {len(self.training_labels)} images:\n"
                            f"• {ripe_count} ripe images\n"
                            f"• {not_ripe_count} not ripe images\n\n"
                            f"You can now test the model to see how well it predicts ripeness!")
        self.info_text.config(state=tk.DISABLED)

    def start_testing(self):
        if not self.model_trained:
            messagebox.showwarning("Model Not Trained", "Please train the model first.")
            return

        # Make predictions for testing set
        self.predictions = []
        for file in self.testing_files:
            features = self.extract_features(file)
            prediction = self.classifier.predict([features])[0]
            self.predictions.append(prediction)

        self.is_training = False
        self.current_image_index = 0

        self.title_label.config(text="Testing Phase: Classify Images")
        self.status_label.config(text=f"Testing: Image 1/{len(self.testing_files)}")

        # Update buttons
        self.train_model_button.config(state=tk.DISABLED)
        self.test_button.config(state=tk.DISABLED)

        # Update info
        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(tk.END,
                              "Testing phase: Classify each image and see how your classifications compare with the model's predictions.")
        self.info_text.config(state=tk.DISABLED)

        self.show_current_image()

    def show_results(self):
        # Clear the main frame
        for widget in self.main_frame.winfo_children():
            widget.destroy()

        # Create a canvas with scrollbar for the entire content
        main_canvas = tk.Canvas(self.main_frame)
        main_scrollbar = ttk.Scrollbar(self.main_frame, orient="vertical", command=main_canvas.yview)
        
        # Create the main container that will hold all content
        main_container = ttk.Frame(main_canvas)
        
        # Configure the canvas
        main_canvas.configure(yscrollcommand=main_scrollbar.set)
        
        # Pack the scrollbar and canvas
        main_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        main_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Create the window in the canvas
        main_canvas.create_window((0, 0), window=main_container, anchor="nw", tags="main_container")
        
        # Configure the scroll region when the main container size changes
        def configure_scroll_region(event):
            main_canvas.configure(scrollregion=main_canvas.bbox("all"))
        main_container.bind("<Configure>", configure_scroll_region)

        # Enable mousewheel scrolling
        def on_mousewheel(event):
            main_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        main_canvas.bind_all("<MouseWheel>", on_mousewheel)

        # Add reset button at the top
        top_frame = ttk.Frame(main_container)
        top_frame.pack(fill=tk.X, pady=(10, 20))

        # Style for the reset button
        style = ttk.Style()
        style.configure("Reset.TButton",
                       font=("Arial", 14, "bold"),
                       padding=15)

        reset_button = ttk.Button(
            top_frame, 
            text="↺ Reset Game", 
            command=self.reset_game,
            style="Reset.TButton"
        )
        reset_button.pack(side=tk.RIGHT, padx=20)

        # Add title
        self.title_label = ttk.Label(main_container, text="Classification Results", font=("Arial", 16, "bold"))
        self.title_label.pack(pady=10)

        # Create notebook with tabs for each category
        notebook = ttk.Notebook(main_container)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create frames for each category
        ripe_frame = ttk.Frame(notebook)
        not_ripe_frame = ttk.Frame(notebook)
        comparison_frame = ttk.Frame(notebook)
        stats_frame = ttk.Frame(notebook)

        notebook.add(ripe_frame, text="Predicted Ripe")
        notebook.add(not_ripe_frame, text="Predicted Not Ripe")
        notebook.add(comparison_frame, text="Comparison")
        notebook.add(stats_frame, text="Statistics")

        # Create scrollable frames
        ripe_canvas, ripe_scrollable_frame = self.create_scrollable_frame(ripe_frame)
        not_ripe_canvas, not_ripe_scrollable_frame = self.create_scrollable_frame(not_ripe_frame)
        comparison_canvas, comparison_scrollable_frame = self.create_scrollable_frame(comparison_frame)

        # Display results with images
        ripe_count = 0
        not_ripe_count = 0
        match_count = 0
        mismatch_count = 0

        # Create image grids (5 columns)
        ripe_grid = ttk.Frame(ripe_scrollable_frame)
        ripe_grid.pack(fill="both", expand=True, padx=5, pady=5)
        not_ripe_grid = ttk.Frame(not_ripe_scrollable_frame)
        not_ripe_grid.pack(fill="both", expand=True, padx=5, pady=5)

        ripe_images = []
        not_ripe_images = []

        for idx, (file, pred) in enumerate(zip(self.testing_files, self.predictions)):
            if pred == "ripe":
                ripe_images.append(file)
                ripe_count += 1
            else:
                not_ripe_images.append(file)
                not_ripe_count += 1

            # Track matches for statistics
            user_label = self.testing_labels[idx]
            if user_label == pred:
                match_count += 1
            else:
                mismatch_count += 1

            # Add to comparison tab
            frame = ttk.Frame(comparison_scrollable_frame)
            frame.pack(pady=5, padx=5, fill="x")

            # Show thumbnail image
            img_label = ttk.Label(frame)
            img_label.pack(side="left", padx=5)
            self.show_image(file, img_label, self.thumbnail_size)

            # Calculate match status
            match_status = "Match" if user_label == pred else "Mismatch"

            # Background color based on match status
            if match_status == "Match":
                frame.configure(style="Match.TFrame")
            else:
                frame.configure(style="Mismatch.TFrame")

            # Add comparison info
            text = f"Model: {pred}, You: {user_label} - {match_status}"
            file_label = ttk.Label(frame, text=text)
            file_label.pack(side="left", padx=10)

        # Function to create image grid
        def create_image_grid(parent, images, cols=5):
            # Take at least 7 unique images, repeat if necessary
            unique_images = list(dict.fromkeys(images))
            while len(unique_images) < 7 and images:
                unique_images.extend(images[:7-len(unique_images)])
            
            for idx, file in enumerate(unique_images):
                row = idx // cols
                col = idx % cols
                frame = ttk.Frame(parent)
                frame.grid(row=row, column=col, padx=2, pady=2)
                img_label = ttk.Label(frame)
                img_label.pack()
                self.show_image(file, img_label, (100, 100))

        # Show at least 7 images in each category
        create_image_grid(ripe_grid, ripe_images)
        create_image_grid(not_ripe_grid, not_ripe_images)

        # Create statistics visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Accuracy pie chart
        accuracy = match_count / len(self.testing_files) * 100
        labels = ['Correct', 'Incorrect']
        sizes = [accuracy, 100-accuracy]
        colors = ['#90EE90', '#FFB6C1']
        ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Model Accuracy')

        # Prediction distribution bar chart
        predictions = ['Ripe', 'Not Ripe']
        counts = [ripe_count, not_ripe_count]
        bars = ax2.bar(predictions, counts, color=['#98FB98', '#FFA07A'])
        ax2.set_title('Prediction Distribution')
        ax2.set_ylabel('Number of Images')
        
        # Add value labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom')

        # Set y-axis to start from 0 and end slightly above the maximum count
        max_count = max(counts)
        ax2.set_ylim(0, max_count * 1.15)  # Add 15% padding at the top
        
        # Add horizontal grid lines
        ax2.yaxis.grid(True, linestyle='--', alpha=0.7)
        ax2.set_axisbelow(True)  # Put grid lines behind bars
        
        # Adjust y-axis ticks to show reasonable intervals
        if max_count > 10:
            step = max(1, max_count // 10)  # Show about 10 tick marks
            ax2.yaxis.set_ticks(range(0, int(max_count * 1.15), step))

        # Embed the plot in the statistics tab
        canvas = FigureCanvasTkAgg(fig, master=stats_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Add stats text below the graphs
        stats_text = f"Total images: {len(self.testing_files)} | "
        stats_text += f"Predicted ripe: {ripe_count} | "
        stats_text += f"Predicted not ripe: {not_ripe_count} | "
        stats_text += f"Match rate: {match_count}/{len(self.testing_files)} "
        stats_text += f"({round(accuracy, 1)}%)"

        ttk.Label(stats_frame, text=stats_text, font=("Arial", 12)).pack(pady=10)

        # Create explanation section with label and text
        explanation_frame = ttk.Frame(stats_frame)
        explanation_frame.pack(fill=tk.BOTH, padx=10, pady=10, expand=True)

        explanation_label = ttk.Label(
            explanation_frame, 
            text="Explanation:", 
            font=("Arial", 12, "bold")
        )
        explanation_label.pack(side=tk.LEFT, padx=(0, 10), anchor='n')

        # Create a container frame for the text widget to control its size
        text_container = ttk.Frame(explanation_frame)
        text_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Add machine learning info with increased height and minimum size
        info_text = tk.Text(text_container, height=6, width=65, wrap=tk.WORD)
        info_text.pack(fill=tk.BOTH, expand=True)
        
        # Set minimum size for the text widget
        info_text.config(height=6)  # Enforce minimum height
        
        # Add the explanation text
        info_text.insert(tk.END,
                         f"This model was trained using a Random Forest Classifier with {len(self.training_labels)} training samples.\n\n"
                         f"The classifier analyzes color and texture features extracted from each image to make predictions about ripeness. "
                         f"These features include RGB color values, color ratios, and edge patterns that help distinguish between ripe and unripe fruits.\n\n"
                         f"The model achieved a {round(accuracy, 1)}% agreement rate with your classifications.")
        info_text.config(state=tk.DISABLED)

    def create_scrollable_frame(self, parent):
        # Create canvas and scrollbar for scrollable frame
        canvas = tk.Canvas(parent)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        return canvas, scrollable_frame

    def add_image_to_frame(self, parent_frame, file, idx, prediction):
        frame = ttk.Frame(parent_frame)
        frame.pack(pady=5, padx=5, fill="x")

        # Create a label for the image
        img_label = ttk.Label(frame)
        img_label.pack(side="left", padx=5)

        # Show thumbnail image
        self.show_image(file, img_label, self.thumbnail_size)

        # Add filename and user label if available
        text = f"{file}"
        if self.testing_labels[idx]:
            text += f" (You labeled: {self.testing_labels[idx]})"

        file_label = ttk.Label(frame, text=text)
        file_label.pack(side="left", padx=10)

    def reset_game(self):
        """Completely reset the game and restart the application"""
        self.root.destroy()
        root = tk.Tk()
        style = ttk.Style()
        style.configure("Match.TFrame", background="#e6ffe6")  # Light green
        style.configure("Mismatch.TFrame", background="#ffe6e6")  # Light red
        game = FruitRipenessGame(root)
        root.mainloop()


if __name__ == "__main__":
    root = tk.Tk()
    # Create styles for comparison frames
    style = ttk.Style()
    style.configure("Match.TFrame", background="#e6ffe6")  # Light green
    style.configure("Mismatch.TFrame", background="#ffe6e6")  # Light red
    game = FruitRipenessGame(root)
    root.mainloop()