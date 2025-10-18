"""
Fix IKEA dataset loader for actual IKEA Manuals at Work data format
"""

import sys
from pathlib import Path

# Read current file
dataset_path = Path('datasets/ikea_dataset.py')
with open(dataset_path, 'r') as f:
    content = f.read()

# Fix 1: parts are nested in 'manual' object
old_parts_logic = """                # Try multiple possible keys for parts
                parts = None
                for key in ['manual_parts', 'added_parts', 'parts', 'part_ids', 'components']:
                    if key in step_data:
                        parts = step_data[key]
                        break"""

new_parts_logic = """                # Try multiple possible keys for parts
                parts = None

                # CRITICAL: Real IKEA data has parts nested in 'manual' object
                if 'manual' in step_data and isinstance(step_data['manual'], dict):
                    manual_obj = step_data['manual']
                    if 'parts' in manual_obj:
                        # parts is a list of part IDs like ['0', '2', '1']
                        part_ids = manual_obj['parts']
                        # Convert to expected format
                        parts = [{'part_id': pid, 'instance_id': f'{pid}_0'} for pid in part_ids]

                # Fallback to original keys if not found
                if parts is None:
                    for key in ['manual_parts', 'added_parts', 'parts', 'part_ids', 'components']:
                        if key in step_data:
                            parts = step_data[key]
                            break"""

content = content.replace(old_parts_logic, new_parts_logic)

# Fix 2: manual_connections also nested
old_conn_logic = """                # Get manual connections if available
                manual_connections = step_data.get('manual_connections', [])"""

new_conn_logic = """                # Get manual connections if available
                manual_connections = []
                if 'manual' in step_data and isinstance(step_data['manual'], dict):
                    manual_obj = step_data['manual']
                    # Try both 'connections' and 'connnections' (typo in some data)
                    manual_connections = manual_obj.get('connections', manual_obj.get('connnections', []))

                # Fallback
                if not manual_connections:
                    manual_connections = step_data.get('manual_connections', [])"""

content = content.replace(old_conn_logic, new_conn_logic)

# Fix 3: Force image resize to exact size
old_load_image = """    def _load_image(self, image_path: Path) -> np.ndarray:
        \"\"\"Load and preprocess manual image\"\"\"
        if image_path.exists():
            image = Image.open(image_path).convert('RGB')
        else:
            # Create a placeholder image if file doesn't exist
            warnings.warn(f"Image not found: {image_path}, using placeholder")
            image = Image.new('RGB', self.image_size, color='white')

        # Resize to target size (force to exact size)
        image = image.resize(self.image_size[::-1], Image.BILINEAR)  # PIL uses (W, H)

        # Convert to numpy
        image = np.array(image)

        return image"""

new_load_image = """    def _load_image(self, image_path: Path) -> np.ndarray:
        \"\"\"Load and preprocess manual image\"\"\"
        if image_path.exists():
            image = Image.open(image_path).convert('RGB')
        else:
            # Create a placeholder image if file doesn't exist
            warnings.warn(f"Image not found: {image_path}, using placeholder")
            image = Image.new('RGB', self.image_size, color='white')

        # FORCE resize to exact target size
        # image_size is (H, W), PIL uses (W, H)
        target_w, target_h = self.image_size[1], self.image_size[0]
        image = image.resize((target_w, target_h), Image.BILINEAR)

        # Convert to numpy
        image = np.array(image)

        # Verify size after conversion
        assert image.shape[:2] == (target_h, target_w), f"Image size mismatch: {image.shape[:2]} != ({target_h}, {target_w})"

        return image"""

content = content.replace(old_load_image, new_load_image)

# Save
with open(dataset_path, 'w') as f:
    f.write(content)

print("✓ Fixed IKEA dataset loader!")
print("  - Parts now extracted from 'manual.parts'")
print("  - Connections from 'manual.connections'")
print("  - Image resize forced to exact size")
