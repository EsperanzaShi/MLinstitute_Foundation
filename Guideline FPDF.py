from fpdf import FPDF

class PDF(FPDF):
    def header(self):
        # Title at the top of each page
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'Foundation Project Guidelines', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        # Footer with page number
        self.set_y(-15)
        self.set_font('Arial', 'I', 10)
        self.cell(0, 10, 'Page ' + str(self.page_no()), 0, 0, 'C')

def add_paragraph(pdf, text):
    # Encode text to latin-1, replacing characters that cannot be encoded
    safe_text = text.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 10, safe_text)
    pdf.ln(2)

# Create instance of PDF class and add a page
pdf = PDF()
pdf.add_page()
pdf.set_font("Arial", size=12)

# Overview Section
add_paragraph(pdf, "Overview: Building an end-to-end application on a self-managed server is an excellent way to prepare for the programme. "
                     "The foundation project is a simplified version of the weekly projects, and involves building, containerizing, and deploying an MNIST digit classifier.")

# Project Brief
add_paragraph(pdf, "Project Brief: Build, containerize, and deploy a simple digit recognizer trained on the MNIST dataset.")

# Section 1: Get Comfortable with the Basics
pdf.set_font("Arial", "B", 12)
pdf.cell(0, 10, "1. Get Comfortable with the Basics", 0, 1)
pdf.set_font("Arial", size=12)
add_paragraph(pdf, "a. Python & PyTorch: Learn Python programming, how to set up virtual environments, and the basics of PyTorch. "
                     "You'll use PyTorch to load the MNIST dataset, build a simple model, train it, and save the weights. "
                     "For detailed tutorials, visit:")
pdf.set_text_color(0, 0, 255)
pdf.write(7, "PyTorch Tutorials", "https://pytorch.org/tutorials/")
pdf.set_text_color(0, 0, 0)
pdf.ln(10)

add_paragraph(pdf, "b. Streamlit for Web Apps: Streamlit enables rapid development of interactive web apps in Python. "
                     "It will be used to create the front-end where users can draw digits or upload images to get predictions. "
                     "Check out the documentation at:")
pdf.set_text_color(0, 0, 255)
pdf.write(7, "Streamlit Docs", "https://docs.streamlit.io/")
pdf.set_text_color(0, 0, 0)
pdf.ln(10)

# Section 2: Train Your MNIST Model Locally
pdf.set_font("Arial", "B", 12)
pdf.cell(0, 10, "2. Train Your MNIST Model Locally", 0, 1)
pdf.set_font("Arial", size=12)
add_paragraph(pdf, "Develop a Python script or Jupyter Notebook that performs the following: "
                     "loading the MNIST dataset (using torchvision), building and training a simple model using PyTorch, and saving the trained model weights. "
                     "You can experiment with the model training in a Jupyter Notebook for immediate feedback and visualization.")

# Section 3: Build the Streamlit Front-End
pdf.set_font("Arial", "B", 12)
pdf.cell(0, 10, "3. Build the Streamlit Front-End", 0, 1)
pdf.set_font("Arial", size=12)
add_paragraph(pdf, "Create a web interface using Streamlit. This interface should include a canvas or image upload area for digit input, "
                     "display the model's prediction and its confidence score, and allow users to provide the correct label for feedback.")

# Section 4: Logging with PostgreSQL
pdf.set_font("Arial", "B", 12)
pdf.cell(0, 10, "4. Logging with PostgreSQL", 0, 1)
pdf.set_font("Arial", size=12)
add_paragraph(pdf, "Store each prediction made by the model, along with the user-provided true label and a timestamp, "
                     "in a PostgreSQL database. This step involves setting up the database, creating the necessary tables, and writing Python scripts to handle the logging. "
                     "For more information on PostgreSQL, visit:")
pdf.set_text_color(0, 0, 255)
pdf.write(7, "PostgreSQL Docs", "https://www.postgresql.org/docs/")
pdf.set_text_color(0, 0, 0)
pdf.ln(10)

# Section 5: Containerization with Docker
pdf.set_font("Arial", "B", 12)
pdf.cell(0, 10, "5. Containerization with Docker", 0, 1)
pdf.set_font("Arial", size=12)
add_paragraph(pdf, "Containerize your application by setting up Docker containers for the PyTorch model/service, the Streamlit web app, and the PostgreSQL database. "
                     "Write Dockerfiles for each service and use Docker Compose to orchestrate them. Further details can be found at:")
pdf.set_text_color(0, 0, 255)
pdf.write(7, "Docker Docs", "https://docs.docker.com/get-started/")
pdf.set_text_color(0, 0, 0)
pdf.ln(10)

# Section 6: Deployment
pdf.set_font("Arial", "B", 12)
pdf.cell(0, 10, "6. Deployment", 0, 1)
pdf.set_font("Arial", size=12)
add_paragraph(pdf, "Deploy your containerized application on a self-managed server (e.g., a Hetzner basic instance, DigitalOcean, or AWS Lightsail). "
                     "Ensure Docker and Docker Compose are installed, open the necessary ports, and run your containers. "
                     "Access the application through a public IP or domain.")

# Section 7: Managing Project Files and Tools
pdf.set_font("Arial", "B", 12)
pdf.cell(0, 10, "7. Managing Project Files and Tools", 0, 1)
pdf.set_font("Arial", size=12)
add_paragraph(pdf, "A key part of this project is organizing your code and files efficiently. Here’s a recommended structure and tool usage:")
add_paragraph(pdf, "Project Folder Structure:")
add_paragraph(pdf, "mnist_digit_classifier/ \n├── docs/                # Documentation and guidelines \n├── notebooks/           # Jupyter notebooks for experimentation \n├── src/                 # Source code organized as follows: \n│   ├── model/           # PyTorch model training and inference scripts \n│   ├── webapp/          # Streamlit application code \n│   └── database/        # Database interaction and logging scripts \n├── docker/              # Dockerfiles and docker-compose.yml \n├── requirements.txt     # Python dependencies \n└── README.md            # Project overview and instructions")
add_paragraph(pdf, "Tools to Use:")
add_paragraph(pdf, "Visual Studio Code: Use this IDE to develop and manage your main project files. Write and organize your Python scripts, Docker configurations, and documentation here. "
                     "Visual Studio Code supports working with multiple files and integrated version control via Git.")
add_paragraph(pdf, "Jupyter Notebook: Use notebooks for prototyping and experimenting, especially when training the MNIST model. "
                     "They provide interactive code execution and visualization, which is ideal for testing different model parameters.")
add_paragraph(pdf, "GitHub: Use Git for version control and push your project to GitHub to keep track of changes, collaborate if needed, and back up your work. "
                     "Integrate with Visual Studio Code through Git extensions for seamless version control.")

# Section 8: Add Project to GitHub
pdf.set_font("Arial", "B", 12)
pdf.cell(0, 10, "8. Add Project to GitHub", 0, 1)
pdf.set_font("Arial", size=12)
add_paragraph(pdf, "Once your project structure is set up and your code is written, commit your changes and push your project to a GitHub repository. "
                     "Include all project files (code, Dockerfiles, notebooks, and documentation) in the repository, and update your README.md with instructions on how to run and deploy your application.")

# Final Summary
pdf.set_font("Arial", "B", 12)
pdf.cell(0, 10, "Summary", 0, 1)
pdf.set_font("Arial", size=12)
add_paragraph(pdf, "This guide outlines the process of building an end-to-end MNIST digit recognizer. "
                     "It covers training a PyTorch model, creating an interactive Streamlit web app, logging predictions in PostgreSQL, containerizing the application with Docker, and finally deploying it on a self-managed server. "
                     "Furthermore, it provides guidance on how to manage your project files using Visual Studio Code for main development, Jupyter Notebook for prototyping, and GitHub for version control. "
                     "Follow these detailed steps to build and deploy your project successfully.")

# Save the PDF with clickable links
output_filename = "Foundation_Project_Guidelines.pdf"
pdf.output(output_filename)
print(f"PDF created successfully: {output_filename}")