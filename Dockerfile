# atiladv DOCKERFILE

# BASE IMAGE
FROM debian:bookworm AS base

# General configs
ENV CONTAINER_USER="atiladv"  
ENV WORKDIR_PATH="/usr/atiladv/atiladv"  
ENV PYTHON_UV_PATH="/python"  
ENV TZ="America/Sao_Paulo"
ENV PROJECT_PYTHON_VERSION="==3.13.*" 
ENV WSCP_DOWNLOADS_ROOT_FOLDER="/var/atiladv/"


# Non-root user
RUN adduser --disabled-password --gecos "" ${CONTAINER_USER}
RUN mkdir -p ${WORKDIR_PATH} ${PYTHON_UV_PATH}
RUN chown -R ${CONTAINER_USER} ${WORKDIR_PATH} ${PYTHON_UV_PATH}
RUN mkdir -p /tmp/atiladv /var/atiladv  
RUN chown -R ${CONTAINER_USER} /tmp/atiladv /var/atiladv

USER ${CONTAINER_USER}
WORKDIR ${WORKDIR_PATH}
 

# PRODUCTION
FROM base AS production


# Setting up the python environment with uv
# See: https://docs.astral.sh/uv/guides/integration/docker
# See: https://github.com/astral-sh/uv-docker-example?tab=readme-ov-file
# uv set must be in sync with pyproject.toml
COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /uvx /bin/ 
ENV UV_PYTHON_INSTALL_DIR=${PYTHON_UV_PATH}
ENV UV_PYTHON_PREFERENCE=only-managed
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
# # Install the project's dependencies 
RUN uv python install ${PROJECT_PYTHON_VERSION} 
# # Installing source code separately from its dependencies to optimize layer caching
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --no-install-project --compile-bytecode

# # Add the venv created by uv to the PATH to "expose" the environment inside container
ENV PATH="${WORKDIR_PATH}/.venv/bin:$PATH"
# # Copy and Sync atiladv source code (I couldn't resolve the sync. So do it manually)
COPY . ${WORKDIR_PATH}
# # Uncomment after first build.
# RUN uv sync --compile-bytecode

CMD ["bash"]


# DEVELOPMENT
FROM production AS dev

USER root
RUN apt update && apt install git -y && apt clean
USER ${CONTAINER_USER}