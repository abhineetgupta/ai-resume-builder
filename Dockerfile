ARG BASE_CONTAINER=jupyter/minimal-notebook:python-3.11
FROM $BASE_CONTAINER

# Install following Python 3 packages with pip
RUN pip install --no-cache-dir \
    'black' \
    'isort' \
    'langchain' \
    'openai' && \
    mamba clean --all -f -y && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"
