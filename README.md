# capstone2-soccer-prediction
A machine learning project focused on predicting soccer match outcomes using historical match data and team performance statistics. The repository includes data collection, preprocessing, exploratory analysis, feature engineering, and predictive modeling workflows.

## Environment Setup

### 1. Create a Virtual Environment

From the project root directory:

```bash
python3 -m venv .venv
```

### 2. Activate the Virtual Environment

**macOS / Linux**

```bash
source .venv/bin/activate
```

**Windows**

```bash
.venv\Scripts\activate
```

After activation, your terminal should display:

```bash
(.venv)
```

### 3. Install Project Dependencies

```bash
pip install -r requirements.txt
```

### 4. Register the Jupyter Kernel

```bash
python -m ipykernel install --user --name capstone2-soccer --display-name "Python (capstone2-soccer)"
```

### 5. Select the Kernel in VS Code

Open any notebook and select:

```text
Python (capstone2-soccer)
```

### Updating Dependencies

If new packages are added to the project, update the dependency file:

```bash
pip freeze > requirements.txt
```