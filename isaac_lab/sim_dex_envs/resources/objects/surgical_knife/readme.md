taken the original obj (after centerign it with MeshLab) and reduced the number of faces with MeshLab usign `filters -> Remeshing, Simplification and Reconstruction ->  Simplification: Quadratic Edge Collapse Decimation`

then /isaaclab.sh -p scripts/tools/convert_mesh.py /home/max/sim_dex_envs/resources/objects/surgical_knife/surgical_knife.obj /home/max/sim_dex_envs/resources/objects/surgical_knife/surgical_knife.usd --mass 0.1 --collision-approximation sdf  --rotate 180 0 -90 --scale 0.18
