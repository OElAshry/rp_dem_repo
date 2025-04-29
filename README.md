The "Meshes" folder contains all the meshes that were used for all 180 simulations (including 5x screw spacings and 3x screw flights). 

The "Inputs" folder contains the LIGGGHTS scripts that were used to run the simulations. Within the MCC and Binary script names, the parameters being simulated were encoded. For example:
1. 5PD - represents screw spacing (originally used PD for particle diameter as that was the initial assumption, but actual spacings are in PR i.e. 5PD actually means 5PR).
2. 1.5N - represents screw flights (where N = 20 flights, so 1.5N = 30 flights)
3. 600 - represents 600 RPM
4. r6 - represents radius value used in conjunction with r0 (BINARY EXCLUSIVE)

The Python script used to generate and submit the simulation script has also been included within the Inputs folder.
