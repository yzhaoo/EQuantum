import bpy
import json
from mathutils import Vector

def export_setup(filename="setup_config.json"):
    config_data = {"regions": []}
    
    for obj in bpy.context.scene.objects:
        # We assume that the objects representing components are mesh objects.
        if obj.type != 'MESH':
            continue
        
        # Use a custom property "component" if available, otherwise default to the object name.
        component = obj.get("component", obj.name)
        
        # Get the object's bounding box in world coordinates.
        # The obj.bound_box property gives 8 local space corners; we transform them to world space.
        coords = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
        xs = [p.x for p in coords]
        ys = [p.y for p in coords]
        zs = [p.z for p in coords]
        bbox = {
            "xmin": min(xs),
            "xmax": max(xs),
            "ymin": min(ys),
            "ymax": max(ys),
            "zmin": min(zs),
            "zmax": max(zs)
        }
        
        # Optionally, you can assign other properties as custom properties on the object.
        # For example, you might set custom properties in Blender like "charge", "potential",
        # "dielectric_constant", and "BCtype" (boundary condition type).
        region_data = {
            "name": component,
            "bbox": bbox,
            "charge": obj.get("charge", 0.0),
            "potential": obj.get("potential", 0.0),
            "dielectric_constant": obj.get("dielectric_constant", 1.0),
            "BCtype": obj.get("BCtype", "n")
        }
        config_data["regions"].append(region_data)
    
    # Write the configuration to a JSON file.
    with open(filename, 'w') as f:
        json.dump(config_data, f, indent=4)

# Run the export function:
export_setup("setup_config.json")