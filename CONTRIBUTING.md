# Contributing to MapfBench

## Getting Started
1. Fork the repository
2. Clone your fork
3. Create a virtual environment
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
4. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

## Development Workflow
1. Create a new branch for your feature
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Make your changes
3. Run tests
   ```bash
   pytest tests/
   ```

## Commit Guidelines
- Use clear, descriptive commit messages
- Separate subject from body with a blank line
- Use imperative mood ("Add feature" not "Added feature")
- Limit first line to 50 characters
- Reference issues and pull requests when applicable

## Code Style
- Follow PEP 8 guidelines
- Use type hints
- Write docstrings for all functions and classes
- Maintain consistent code formatting

## Submitting a Pull Request
1. Push your changes to your fork
2. Create a pull request with a clear description of changes
3. Ensure all tests pass
4. Await code review
