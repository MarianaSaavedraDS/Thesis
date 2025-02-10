# Thesis

### **Setting Up the Development Environment**

#### **1. Clone the Repository**
```bash
git clone https://github.com/yourusername/yourrepo.git
cd yourrepo
```

#### **2. Create and Activate the Virtual Environment**
You can create and activate a virtual environment to manage your dependencies:

##### For Windows:
```bash
python -m venv .venv
.venv\Scripts\activate
```

#### **3. Install Dependencies**
Once the virtual environment is activated, install the necessary dependencies:

```bash
pip install -r requirements.txt
```

#### **4. Install the Local Package**
In the project root directory, install your package locally to enable imports from `libs/`:

```bash
pip install -e .
```

This ensures that the `libs/` folder is treated as a Python package.
