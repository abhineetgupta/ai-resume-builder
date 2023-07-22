# ai-resume-builder
Create $\LaTeX$ and `pdf` résumés that have been tailored to each individual job posting by an AI LLM model

### Installation
1. Clone this repo.
1. Provide argument values in `env` file. Instructions are in the file. Save the file as `.env`.
1. Open Docker. Install from [docker.com](https://www.docker.com/products/docker-desktop/) if not available.
1. Open `terminal` and `cd` into the cloned repo.
1. Start the jupyterlab notebook server - 
    ```
    docker-compose up --build -d
    ```
1. Open this URL in your browser to access the notebook - [http://127.0.0.1:8888/lab](http://127.0.0.1:8888/lab). If you had specififed a different port in the `.env` file, change it accordingly.
1. Open the notebook - `resume_builder.ipynb`, and follow the steps within the notebook to create tailored résumés.
1. When done, optionally turn down the notebook server -
    ```
    docker-compose down -v
    ```

### How it works
1. A raw résumé is provided by the user as a `yaml` file, which includes all their basic information, experience, projects, and skills. This file will be used as a basis for populating the final résumé corresponding to each job post. An example file `raw_resume_example.yaml` is included with steps for creating your own.
1. I have created LLM prompts in `prompts.py` for different résumé related tasks, such as, parsing a job post, rewriting a résumé section, extracting skills, identifying improvements, and writing a summary. The prompts are tested using ChatOpenAI models from [Langchain](https://js.langchain.com/docs/api/chat_models_openai/classes/ChatOpenAI).
1. Based on a job post, the user can perform some or all the above tasks to generate a new `yaml` file with résumé data that is the most relevant for the job opportunity.
1. The `yaml` file is then converted to $\LaTeX$ using a `jinja2` template. Templates are provided in the `templates` directory.
1. Finally, the $\LaTeX$ file is converted to `pdf` using the `texlive` distribution installed in the `Dockerfile`.

### Personal file folder
This github repo is configured to not track any directory starting with `my_`, so these directories can be used to store personal data without inadvertantly pushing it to github.

___ 
Suggestions are welcome. Please create an issue, or fork and PR if you would like to share any improvements in the code or prompts.