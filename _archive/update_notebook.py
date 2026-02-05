import json
import os

notebook_path = "pizza_stream_pipeline.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

# Helper to find cell by source content (approximate)
def find_cell_index(cells, content_snippet):
    for i, cell in enumerate(cells):
        if cell['cell_type'] == 'code':
             source = "".join(cell['source'])
             if content_snippet in source:
                 return i
    return -1

# 1. Update import cell
idx_import = find_cell_index(cells, "import utils as u")
if idx_import != -1:
    cells[idx_import]['source'] = [
        "import utils_pizzastream as u\n",
        "import os\n",
        "# Load configuration\n",
        "config = u.load_config()"
    ]

# 2. Update paths cell
idx_paths = find_cell_index(cells, "video_original = rf'C:\\Users\\Franki\\Videos")
if idx_paths == -1:
     idx_paths = find_cell_index(cells, "video_original =") # Fallback
     
if idx_paths != -1:
    cells[idx_paths]['source'] = [
        "# Link del video o ruta local\n",
        "video_original = os.path.join(config['videos_input_dir'], f\"{ps_nn}.mp4\")\n",
        "\n",
        "base_dir = os.path.join(config['vault_path'], config['project_root'])\n",
        "video_editado = os.path.join(base_dir, ps_nn, f\"{ps_nn}.mp4\")"
    ]

# 3. Update base_dir cell (now redundant but let's just comment it out or update it)
idx_base = find_cell_index(cells, "base_dir = r\"G:\\Mi unidad")
if idx_base != -1:
    cells[idx_base]['source'] = [
        "# base_dir is defined in the previous cell from config\n",
        "# base_dir = ... (replaced by config)"
    ]

# 4. Update function call cell
idx_func = find_cell_index(cells, "u.crear_carpeta_y_obsidian")
if idx_func != -1:
    cells[idx_func]['source'] = [
        "current_folder = u.crear_carpeta_y_obsidian(ps_nn,\n",
        "                                            vault_path = config['vault_path'],\n",
        "                                            root_folder = config['project_root']\n",
        "                                            )"
    ]

# Write back
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook updated successfully.")
