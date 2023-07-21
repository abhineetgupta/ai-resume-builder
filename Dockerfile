ARG BASE_CONTAINER=jupyter/minimal-notebook:python-3.11
FROM $BASE_CONTAINER

USER root

# Texlive makes it impossible to install packages later, so just bite the bullet and install texlive-full without the beef
RUN apt-get update --yes && \
    apt-get install --yes --no-install-recommends \
    `apt-get --assume-no install texlive-full | \
		awk '/The following additional packages will be installed/{f=1;next} /Suggested packages/{f=0} f' | \
		tr ' ' '\n' | \
        grep -vP 'doc$' | \
        grep -vP 'texlive-lang' | \
        grep -vP 'latex-cjk' | \
        tr '\n' ' '`

# Switch back to jovyan to avoid accidental container runs as root
USER ${NB_UID}

# Install following Python 3 packages with pip
RUN pip install --no-cache-dir \
    'black' \
    'isort' \
    'langchain' \
    'openai' && \
    mamba clean --all -f -y && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"
