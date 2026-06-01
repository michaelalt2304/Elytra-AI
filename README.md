# Elytra Robotics AI Platform

Welcome to the Elytra Robotics AI Platform repository.

If you have questions about setup, usage, or development, please contact **[mjstraus2304@gmail.com](mailto:mjstraus2304@gmail.com)**.

## Important Notes

* Video (`.mp4`) files are excluded from this repository via `.gitignore` and are not distributed through GitHub.
* To run the project, download or create a sample video containing beach litter/trash footage and place it in the appropriate project folder.
* Update the video file path in `main.py` to point to your local video file.

---

## Getting Started

### 1. Clone the Repository

```bash
git clone <repository-url>
cd <repository-name>
```

### 2. Install Python 3.8.20

Download and install Python 3.8.20:

https://www.python.org/downloads/release/python-3820/

### 3. Add Python to Your PATH

Ensure that the installation directory containing `python.exe` is added to your system `PATH`.

You can verify your installation by running:

```bash
python --version
```

Expected output:

```bash
Python 3.8.20
```

### 4. Verify Python Configuration

If you are using a different Python installation than the default one in your system `PATH`, update the Python executable path specified in `run_me.py`.

### 5. Run the Project

From the repository root directory:

```bash
python run_me.py
```

---

## Troubleshooting

If you encounter setup or dependency errors:

1. Delete the `virtual` directory.
2. Re-run:

```bash
python run_me.py
```

This will recreate the virtual environment and reinstall required dependencies.

Before investigating further issues, always try this step first.

---

## Development Workflow

* Always launch the application using:

```bash
python run_me.py
```

* Make modifications in `main.py`.
* Refer to the source code for advanced configuration options.
* Additional setup notes and troubleshooting guidance can be found in the comments within `run_me.py`.

---

## Known Issues

* A warning may appear during annotation workflows.
* This warning is a known issue and can currently be ignored.
* A fix is planned for a future update.

---

Happy coding! 🚀
