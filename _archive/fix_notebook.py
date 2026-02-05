import json
import os

notebook_path = r"g:\Mi unidad\Obsidian\0_SourceMaterial\El pizzas\1_current\code\notebooks\pizza_stream_pipeline.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find the first code cell (execution_count 1 or id 892b9508)
# Based on previous view_file, it's the first cell
cell = nb['cells'][0]

new_source = [
    "import sys\n",
    "import os\n",
    "\n",
    "# Add the utils directory to sys.path\n",
    "# Assuming notebook is in 'notebooks' and utils_pizzastream is in '../utils_pizzastream'\n",
    "utils_dir = os.path.abspath(os.path.join(os.getcwd(), '..', 'utils_pizzastream'))\n",
    "if utils_dir not in sys.path:\n",
    "    sys.path.append(utils_dir)\n",
    "\n",
    "import utils_pizzastream as u\n",
    "# Load configuration\n",
    "config = u.load_config()"
]

cell['source'] = new_source

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook updated successfully.")
