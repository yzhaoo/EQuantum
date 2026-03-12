# Configuration
# TODO: Set the path to your Blender executable here
BLENDER_PATH="blender" 
BLEND_FILE="setup/congif1.blend" 
SCRIPT_FILE="setup/assign_mat_optimized.py"
OUTPUT_FILE="setup/setup_config1.json"

# Check if Blender executable exists or is in PATH
if ! command -v "$BLENDER_PATH" &> /dev/null && [ ! -x "$BLENDER_PATH" ]; then
    echo "Error: Blender executable not found at '$BLENDER_PATH'."
    echo "Please edit this script and set BLENDER_PATH to the correct location."
    echo "Example: BLENDER_PATH='/home/user/blender-3.6.0-linux-x64/blender'"
    exit 1
fi

if [ ! -f "$BLEND_FILE" ]; then
    echo "Error: Blend file '$BLEND_FILE' not found."
    exit 1
fi

echo "Running material assignment using: $BLENDER_PATH"
"$BLENDER_PATH" "$BLEND_FILE" --background --python "$SCRIPT_FILE"

if [ $? -eq 0 ]; then
    echo "Success! Output generated at $OUTPUT_FILE"
else
    echo "Blender execution failed."
    exit 1
fi
