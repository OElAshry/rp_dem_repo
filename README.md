# rp_liggghts_templates

The "Meshes" folder contains all the meshes that were used for all 180 simulations (including 5x screw spacings and 3x screw). 

The "Inputs" folder contains the template LIGGGHTS scripts that were used to run the simulations. Some things to note:
1. Within the template_MCC.liggghts and template_binary.liggghts, placeholder strings were placed to allow iteration of the parameter space.
2. XXX placeholder - represents screw spacing
3. YYY placeholder - represents screw pitch
4. ZZZ placeholder - represents RPM
5. AAA placeholder - represents radius value used in conjunction with r0 (binary exclusive)

The Python script used to generate and submit the simulation script has also been included within the Inputs folder.
