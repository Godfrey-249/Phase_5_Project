import nbformat

notebook_path = "/Users/faithkamande/Phase_5_Project/Mental_Health_Identifier..ipynb"

def fix_widget_metadata(notebook_path, output_path="/Users/faithkamande/Phase_5_Project/Mental_Health_Identifier2.ipynb"):
    # Load notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=nbformat.NO_CONVERT)

    # Check for broken widget metadata
    widgets = nb['metadata'].get('widgets')
    if widgets and 'application/vnd.jupyter.widget-state+json' in widgets:
        widget_data = widgets['application/vnd.jupyter.widget-state+json']
        if 'state' not in widget_data:
            print("Fixing missing 'state' key in metadata.widgets...")
            widget_data['state'] = {}  # Add empty state if missing
            # Optional: remove stale data if any
            widget_data.pop('version_major', None)
            widget_data.pop('version_minor', None)

    # Save the fixed notebook
    if not output_path:
        output_path = notebook_path  # overwrite original

    with open(output_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

    print(f"Notebook fixed and saved to {output_path}")

# === USAGE ===
fix_widget_metadata("Mental_Health_Identifier.ipynb")