# Team_Challenge_SP17_T03
Team Challenge del S17 - Pipelines

Team Members: Antonio Carreño, Jose Ramón Casas, Sergio Risueño, María risco

This repository includes data, .py files, notebooks, and the developed model to predict whether a person, based on specific characteristics, will subscribe to a financial product (a fixed-term deposit).

The files are organized as follows::

Team_Challenge_SP17_T03/
│── env/                    # Virtual environment 
│── src/                    # Main source code directory
│   │── data/               # Contains datasets for training and testing
│   │── models/             # Stores machine learning models
│   │── notebooks/          # Stores output notebooks with results from experiments
│   │── result_notebooks/   # Jupyter notebooks for exploratory data analysis or model development
│   │── utils/              # Utility scripts for shared functionality
│── .gitignore              # Specifies files and folders to be ignored by Git
│── environment.yml         # Conda environment file with dependencies
│── README.md               # Project documentation and setup instructions


The virtual environment defined by the environment.yml file is built using Conda and relies on the conda-forge and defaults channels for package management. It is configured for Python 3.9 and includes a comprehensive set of dependencies for data science, machine learning, and development.

How to Download the Repository and Run the Code
Follow these steps to download the repository and run the code:

1. Clone the Repository
git clone https://github.com/mariarisco/Team_Challenge_SP17_T03.git

2. Navigate to the Project Directory
cd Team_Challenge_SP17_T03

3. Set Up the Virtual Environment
Ensure you have Conda installed, then create and activate the environment:
conda env create -f environment.yml
conda activate env **Check the list of environments with: conda env list and use the name displayed in conda env list.

4. Run the Code
Navigate to the src/result_notebooks directory.
Open and execute Team_Pipelines_I.ipynb by running all the cells in order. This will train and save the best-performing ML model.

If you want to test the model, open and execute Team_Pipelines_II.ipynb.

5. Deactivate the Environment When Done
conda deactivate

