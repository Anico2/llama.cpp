FROM python:{{cookiecutter.python_version }}-slim

# Install uv for fast dependency management
RUN pip install uv

WORKDIR /app

# Copy dependency files
COPY pyproject.toml .

# Install dependencies using uv
RUN uv pip install --system -e .

# Copy the rest of the application
COPY . .

# Expose the ADK API port
EXPOSE 8000

CMD ["python", "src/{{ cookiecutter.package_name }}/infrastructure/api/main.py"]
