import bpy
import json
import mathutils
from mathutils.bvhtree import BVHTree

def import_sites(filename="sites.json"):
    with open(filename, "r") as f:
        sites_data = json.load(f)
    # Convert the site data into a list of mathutils.Vector objects (or dictionaries if you need full properties)
    sites = []
    for site in sites_data:
        coord = mathutils.Vector(site["coordinates"])
        # You could also store the entire dictionary if you need material and other properties
        sites.append({
            "id": site["id"],
            "coordinates": coord,
            "material": site["material"],
            "charge": site["charge"],
            "potential": site["potential"],
            "dielectric_constant": site["dielectric_constant"],
            "BCtype": site["BCtype"]
        })
    return sites


def is_point_inside_object(point, obj, depsgraph, site=None):
    """
    Test if a point is inside a closed mesh object.
    
    If a point is found to be inside and a Site object is provided via 'site',
    update the site's properties from the object's material or custom properties.
    
    Parameters:
      point (Vector): The point in world coordinates.
      obj (Object): The Blender object (assumed to be a mesh).
      depsgraph: The dependency graph (e.g. from bpy.context.evaluated_depsgraph_get()).
      site (optional): A Site object to update if the point is inside.
      
    Returns:
      True if the point is inside the mesh, False otherwise.
    """
    # Get the evaluated object (to apply modifiers, etc.)
    eval_obj = obj.evaluated_get(depsgraph)
    mesh = eval_obj.to_mesh()
    bvhtree = BVHTree.FromMesh(mesh)
    
    direction = mathutils.Vector((1, 0, 0))  # arbitrary ray direction
    count = 0
    test_point = point.copy()
    epsilon = 1e-5  # small offset
    
    while True:
        result = bvhtree.ray_cast(test_point, direction)
        if result[0] is None:
            break
        hit_location, hit_normal, face_index, distance = result
        count += 1
        test_point = hit_location + direction * epsilon

    eval_obj.to_mesh_clear()
    
    inside = (count % 2) == 1
    
    # If the point is inside and a site was provided, update its properties.
    if inside and site is not None:
        if obj.data.materials:
            mat = obj.data.materials[0]  # using the first material
            site.material = mat.name
            site.charge = mat.get("charge", site.charge)
            site.potential = mat.get("potential", site.potential)
            site.dielectric_constant = mat.get("dielectric_constant", site.dielectric_constant)
            site.BCtype = mat.get("BCtype", site.BCtype)
        else:
            # Fallback to object custom properties if no material is assigned.
            site.material = obj.name
            site.charge = obj.get("charge", site.charge)
            site.potential = obj.get("potential", site.potential)
            site.dielectric_constant = obj.get("dielectric_constant", site.dielectric_constant)
            site.BCtype = obj.get("BCtype", site.BCtype)
    
    return inside
# Example usage in Blender:
sites = import_sites("sites.json")
print("Imported {} sites".format(len(sites)))
# You can now use these sites for further processing, e.g. running the point-in-mesh test.