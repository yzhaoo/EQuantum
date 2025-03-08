import bpy
import json
import mathutils
from mathutils.bvhtree import BVHTree
import bmesh
import os

def is_point_inside_object(point, obj, depsgraph, site=None):
    """
    Test if a point is inside a closed mesh object using a ray-cast and BVH.
    If the point is inside and a site dictionary is provided, update its properties from the object.
    
    Parameters:
      point (Vector): The point in world coordinates.
      obj (Object): A Blender mesh object.
      depsgraph: Blender's dependency graph.
      site (dict, optional): A site dictionary whose properties will be updated if the point is inside.
    
    Returns:
      bool: True if the point is inside the mesh, False otherwise.
    """
    # Get the evaluated object (with modifiers applied)
    eval_obj = obj.evaluated_get(depsgraph)
    mesh = eval_obj.to_mesh()
    
    # Build BVH tree in world space (ensuring that transformations, including scale, are applied)
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    for vert in bm.verts:
        vert.co = obj.matrix_world @ vert.co
    bvhtree = BVHTree.FromBMesh(bm)
    bm.free()
    eval_obj.to_mesh_clear()
    
    # Set up ray casting.
    direction = mathutils.Vector((1, 1, 1))
    count = 0
    test_point = point.copy()
    epsilon = 1e-5
    
    while True:
        result = bvhtree.ray_cast(test_point, direction)
        if result[0] is None:
            break
        hit_location, hit_normal, face_index, distance = result
        count += 1
        #print(hit_location)
        # Advance the start point slightly past the hit
        test_point = hit_location + direction * epsilon

    # Clean up the temporary mesh
    eval_obj.to_mesh_clear()
    #print(count)
    inside = (count % 2) == 1  # odd number of intersections -> inside
    
    if inside and site is not None:
        # If the object has a material, update properties from the material's custom properties.
        if obj.data.materials:
            mat = obj.data.materials[0]
            if (mat.name =='dielectric' and site["material"]=="Qsystem"):
                pass
            else:
                site["material"] = mat.name
                site["charge"] = mat.get("charge", site.get("charge", 0.0))
                site["potential"] = mat.get("potential", site.get("potential", 0.0))
                site["dielectric_constant"] = mat.get("dielectric_constant", site.get("dielectric_constant", 1.0))
                site["BCtype"] = mat.get("BCtype", site.get("BCtype", "n"))
    
    return inside
def point_in_bbox(point, bbox):
    """
    Check if a point (Vector) is within a bounding box defined as a dictionary:
       { "xmin": ..., "xmax": ..., "ymin": ..., "ymax": ..., "zmin": ..., "zmax": ... }
    """
    return (bbox["xmin"] <= point[0] <= bbox["xmax"] and
            bbox["ymin"] <= point[1] <= bbox["ymax"] and
            bbox["zmin"] <= point[2] <= bbox["zmax"])

def get_bbox(obj):
    coords = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]
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
    return bbox

def assign_properties_for_2dplane(sites, obj):
    """
    First pass: For each site, check if its coordinate is inside the bounding box of the 2dplane object.
    If so, assign the object's properties to the site.
    """
    # Get the object's evaluated bounding box in world space.
    # The object's bound_box is in local coordinates, so we transform it.
    coords = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]
    xs = [p.x for p in coords]
    ys = [p.y for p in coords]
    zs = [p.z for p in coords]
    bbox = {
        "xmin": min(xs),
        "xmax": max(xs),
        "ymin": min(ys),
        "ymax": max(ys),
        "zmin": 0.,
        "zmax": 0.
    }
    for site in sites:
        if point_in_bbox(site["coordinates"], bbox):
            # Update properties from the 2dplane object.
            mat = obj.data.materials[0]
            site["material"] = mat.name
            site["charge"] = mat.get("charge", site.get("charge", 0.0))
            site["potential"] = mat.get("potential", site.get("potential", 0.0))
            site["dielectric_constant"] = mat.get("dielectric_constant", site.get("dielectric_constant", 1.0))
            site["BCtype"] = mat.get("BCtype", site.get("BCtype", "n"))

def import_sites(filename="sites.json"):
    """
    Import site information from a JSON file.
    Converts coordinate lists to mathutils.Vector objects.
    
    Returns:
      list: A list of dictionaries, each representing a site.
    """
    with open(filename, "r") as f:
        sites_data = json.load(f)
    for site in sites_data:
        # Convert coordinates (list) to a mathutils.Vector for easier manipulation.
        site["coordinates"] = mathutils.Vector(site["coordinates"])
    return sites_data

def export_sites(sites_data, filename="updated_sites.json"):
    """
    Export the updated site information to a JSON file.
    Converts mathutils.Vector coordinates back to lists.
    """
    export_data = []
    for site in sites_data:
        site_copy = site.copy()
        # Convert coordinates back to a plain list.
        if isinstance(site_copy.get("coordinates"), mathutils.Vector):
            site_copy["coordinates"] = list(site_copy["coordinates"])
        export_data.append(site_copy)
    with open(filename, "w") as f:
        json.dump(export_data, f, indent=4)

def main():
    filepath=bpy.data.filepath
    directory=os.path.dirname(filepath)
    sitefile=os.path.join(directory,"sites.json")
    # Get Blender's dependency graph
    depsgraph = bpy.context.evaluated_depsgraph_get()
    
    # Import sites from file (assumed to be exported previously)
    sites = import_sites(sitefile)
    print("Imported {} sites".format(len(sites)))
    
    # Get the 2dplane object.
    plane_obj = bpy.data.objects.get("2dplane")
    
    # First pass: for the 2dplane, use only the bounding box test.
    if plane_obj is not None:
        assign_properties_for_2dplane(sites, plane_obj)
    else:
        print("Warning: '2dplane' object not found in the scene.")
    
    # Second pass: for all other mesh objects (excluding 2dplane),
    # use the BVH method to update properties.
    other_objects = [obj for obj in bpy.context.scene.objects 
                     if obj.type == 'MESH' and obj.name != "2dplane"]
    
    # For each site, test whether it is inside any of the objects.
    # Once a site is found to be inside an object, its properties are updated.
    for site in sites:
        point = site["coordinates"]
        for obj in other_objects:
            bbox=get_bbox(obj)
            if point_in_bbox(site['coordinates'],bbox):
                if is_point_inside_object(point, obj, depsgraph, site=site):
                    # Optionally, break if you want the first matching object to determine properties.
                    break
    
    # Export the updated sites to a new file.
    export_sites(sites, os.path.join(directory,"updated_sites.json"))
    print("Updated site information exported to updated_sites.json")

if __name__ == "__main__":
    main()




