original file from: UNKNOWN

convertd the usd to obj to then call 
./isaaclab.sh -p scripts/tools/convert_mesh.py sim_dex_envs/resources/objects/surgical_scissors/surgical_scissors.obj sim_dex_envs/resources/objects/surgical_scissors/surgical_scissors.usd --mass 0.1 --collision-approximation sdf  --rotate 0 0 -90 --target-dimensions 0.15 0.1 0.005

we then added a pivot to the mesh and moved the mesh to the origin (by visualising the grid and moving the frame there)