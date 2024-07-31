# Laptop Finder

Welcome to **Laptop Finder**, a webapp chatbot designed to provide laptop recommendations based on your budget and requirements. You can access the webapp [here](https://gadget-finder.streamlit.app/).

## Features

- **Personalized Recommendations**: Get laptop suggestions tailored to your specific needs and budget.
- **User-Friendly Interface**: Easy-to-use interface powered by Streamlit.
- **Dynamic Filtering**: Adjust your preferences to receive the most suitable recommendations.

## Getting Started

### Prerequisites

- Python 3.12 
- Streamlit

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/manit2004/laptop-finder.git
    cd laptop-finder
    ```
2. Create and activate a Python virtual environment:
    - **Windows**:
        ```powershell
        python -m venv venv
        .\venv\Scripts\activate
        ```
    - **Linux**:
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4. Create your own .env in the root directory and put your own Groq api key, langchain hub api key, google api key, google cse id. 

## Usage

1. Open the webapp in your browser:
    ```bash
    streamlit run gadgetfinder.py
    ```
2. Interact with the chatbot to receive laptop recommendations based on your input criteria such as budget, preferred brand, usage, etc.

## Project Structure

- `gadgetfinder.py`: The main file that runs the Streamlit webapp.
- `requirements.txt`: List of Python libraries required to run the app.
- `vectorstore`: Directory containing vector database (FAISS).

## Contributing

We welcome contributions! If you have suggestions or improvements, please create a pull request or open an issue.


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Streamlit](https://www.streamlit.io/) for providing an amazing platform to build the webapp.
- All contributors and users for their valuable feedback and support.