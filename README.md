# track_generator
Generates an image with a car track. This can be used to make the grount of 3D race simulations

The circuit_cente generator will use a set of textures to construct a randomized ground image, then it will draw the track on the ground image.
The set of textures are stored in the texture folder.


The circuit array definition is declared in the def_track.json file : 
     - All values in this files are to be set in mm in real world. 
     - tile_width and tile_height are the sizes of the textures ( in real world) that will be used to fill the ground ( 500 = 0,5 m) 
     - defc is the circuit definition : each line describes the start of a portion of track. The last line must match with the first line.
          syntax is as follow : 
           [axis_center_x, axis_center_y, portion_angle_start, Radius1, radius2,trajectory_point, nb_vertices]
           axis_center_x,axis_center_y : reference point which is used to draw the portion. 
           portion_angle_start : angle from the ref point .  0 is the x axis going on the right. positive values will go down ( like a clock).



