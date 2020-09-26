#perspective
from PIL import Image, ImageDraw

#from google.colab.patches import cv2_imshow
#from IPython.display import display # to display images
import json

import matplotlib.pyplot as plt
#import cv2
import numpy as np
import math
from numpy import mean

pi=3.14


# used to generate the images an json files
import os
from docopt import docopt
import json

import random



class circuit():
    def __init__(self,circuit_data=None, track_folder=None, ClearFolder=None,scalefactor=4):
        #### be carefull, the clearfolder option will delete all information from the folder ! 

        self.scalefactor=scalefactor # to save memory we will work with images where one pixel is a cm instead of images where 1pixel=1mm.
                              # 1 pixel = 1 * reducefactor mm

   #    self.visible_surrounding= int(math.sqrt((self.visible_world_width/2)*(self.visible_world_width/2)+self.visible_world_height*self.visible_world_height))+10
        #self.path = os.path.expanduser(path)
        #self.meta_path = os.path.join(self.path, 'meta.json')

        self.BLACK = (0, 0, 0)   
        #self.WHITE = (170, 175, 169)  #used with success.
        self.WHITE = (255, 255, 255)

        self.WHITE_LEFT = (220, 220, 210)
        self.WHITE_RIGHT = (220, 220, 210)
        self.WHITE_CENTER = (220, 220, 210)


        if track_folder is None : 
           self.track_folder="Tracks/Cristallin"
        else : 
           self.track_folder=track_folder

        self.texture_folder=self.track_folder+"/textures"

        self.circuit_info={}
        self.circuit_info["width"]=22000
        self.circuit_info["height"]=8000
        self.circuit_info["altitude"]=2200
        self.circuit_info["tile_width"]=500
        self.circuit_info["tile_height"]=500
        self.circuit_info["translate_x"]=500
        self.circuit_info["translate_y"]=1500

        self.circuit_info["LINE_WIDTH"]=100
        self.circuit_info["COLOR_LEFT"]=(220, 220, 210)
        self.circuit_info["COLOR_RIGHT"]=(220, 220, 210)
        self.circuit_info["COLOR_CENTER"]=(220, 220, 210)

        self.defc=None

        # distances are in mm
        # build default circuit structure : 
        if ClearFolder == True : 

          self.defc1=[
              [1500,1500,-3*pi/2,500,1500,0,0],
              [1500,1500,-pi,500,1500,0,0],
              [1500,3100+1500,-pi,500,1500,0,0],
              [1500,3100+1500,0,500,1500,0,0],
              [3500,2600,-pi,1500,500,0,0],
              [3500,2600,-pi-pi/2,1500,500,0,0], 
              [8000,2600,-pi-pi/2,1500,500,0,0], 
              [8000,2600,-2*pi-pi/2,1500,500,0,0],
              [4550,4600,pi/2,500,1500,0,0], 
              [4550,4600,pi/2+pi,500,1500,0,0], 
              [18500,4600,pi/2+pi,500,1500,0,0],
              [18500,4600,pi+pi,500,1500,0,0],
              [18500,1500,0,500,1500,0,0],
              [18500,1500,pi,500,1500,0,0],
              [16500,3500,0,1500,500,0,0],
              [16500,3500,-pi/2,1500,500,0,0],
              [12000,3500,-pi/2,1500,500,0,0],
              [12000,3500,-pi-pi/2,1500,500,0,0],
              [15450,1500,-pi/2,500,1500,0,0],
              [15450,1500,pi-pi/2,500,1500,0,0]]
          self.defc2=[
              [1500,1500,-3*pi/2,500,1500,0.5,0],
              [1500,1500,-pi/2 +pi/4-0.1, 500,1500,-0.5,1],
              [2765,3050,pi/2+pi/4-0.1,1500,500,-0.5,0],
              [2765,3050,-pi/2-pi/4+0.1,1500,500,0.5,1],
              [1500,4600,-3*pi/2-pi/4+0.1,500,1500,0.5,0],
              [1500,4600,-pi/2,500,1500,0,0],
              [15500,4600,-pi/2,500,1500,0.5,0],
              [15500,4600,pi/2,500,1500,0,0],
              [6500,3550,-pi/2,1500,500,-0.5,0],
              [6500,3550,-pi,1500,500,-0.5,0],
              [6500,2550,-pi,1500,500,-0.5,0],
              [6500,2550,-3*pi/2,1500,500,0,0],
              [15500,2550,-3*pi/2,1500,500,-0.5,0],
              [15500,2550,-3*pi/2-pi/4,1500,500,0,0],
              [19500,4600,-pi/2-pi/4,500,1500,-0.5,0],
              [19500,4600,0,500,1500,0,0],
              [19500,1500,0,500,1500,0.5,0],
              [19500,1500,pi/2,500,1500,0,0]]

          self.defc=self.defc2
          self.circuit_info["defc"]=self.defc
          # save the default circuit
          try:
            test=json.dumps(self.circuit_info)
            test = test.replace("],", "],\n")
            test = test.replace(', "', ', '+"\n"+'"')
            test = test.replace(' "', '"')

            with open(self.track_folder+"/def_track.json", 'w') as fp:
                fp.write(test)
                #json.dump(self.circuit_info, fp,indent=3)
                

          except:
                print("error opening file :", self.track_folder+"/def_track.json")


        # Load circuit info from file : 
        try:
            with open(self.track_folder+"/def_track.json", 'r') as f:
                self.circuit_info = json.load(f)
        except FileNotFoundError:
                print("Error loading track config")

        self.defc=self.circuit_info["defc"]

        #sizes of circuit are in mm.
        self.width,self.height = self.circuit_info["width"],self.circuit_info["height"]
        self.ceiling_altitude = self.circuit_info["altitude"]

        # size of tiles in real world : ( size are in mm)
        self.Matrix_Ceiling_tile_width=self.circuit_info["tile_width"]
        self.Matrix_Ceiling_tile_height=self.circuit_info["tile_height"]
        # how many tiles will be needed to draw the ceiling ?  
        self.Matrix_Ceiling_width=int(self.width/self.Matrix_Ceiling_tile_width)+1
        self.Matrix_Ceiling_height=int(self.height/self.Matrix_Ceiling_tile_height)+1
        self.Matrix_Ceiling = [[0 for x in range(self.Matrix_Ceiling_width)] for y in range(self.Matrix_Ceiling_height)]

        # the floor matrix with only texture numbers, not instance texture numbers : 
        self.Simple_Matrix_Ceiling = [[0 for x in range(self.Matrix_Ceiling_width)] for y in range(self.Matrix_Ceiling_height)]

        # size of tiles in real world : ( size are in mm)
        self.Matrix_Floor_tile_width=self.circuit_info["tile_width"]
        self.Matrix_Floor_tile_height=self.circuit_info["tile_height"]

        self.tile_w, self.tile_h = self.Matrix_Floor_tile_width,self.Matrix_Floor_tile_height


        # how many tiles will be needed to draw the ceiling ?  
        self.Matrix_Floor_width=int(self.width/self.Matrix_Floor_tile_width)+1
        self.Matrix_Floor_height=int(self.height/self.Matrix_Floor_tile_height)+1
        self.Matrix_Floor = [[0 for x in range(self.Matrix_Floor_width)] for y in range(self.Matrix_Floor_height)]

        # the floor matrix with only texture numbers, not instance texture numbers : 
        self.Simple_Matrix_Floor = [[0 for x in range(self.Matrix_Floor_width)] for y in range(self.Matrix_Floor_height)]



        #when building a new map : this will generate dummy map_texture files : 
        if ClearFolder == True : 
            self.init_empty_map() 


        self.translate_circuit(self.circuit_info["translate_x"],self.circuit_info["translate_y"])

         # define the array of available textures : 
        self.txlist=[]
        self.txnumbers=[]

        # define the 4 wall images ( left, top , right, bottom)
        self.side1,self.side1_w, self.side1_h=None,0,0
        self.side2,self.side2_w, self.side2_h=None,0,0
        self.side3,self.side3_w, self.side3_h=None,0,0
        self.side4,self.side4_w, self.side4_h=None,0,0

        #ground tile size

        self.load_textures()  
        self.load_map()  
 

        # build the thumbnail image of the circuit : 
        self.basewidth = int(self.width/self.scalefactor)
        self.wpercent = (1/self.scalefactor)

        hsize = int((float(self.height)/self.scalefactor))

        self.thumbnailcircuit= Image.new('RGB', (self.basewidth,hsize), self.BLACK)
        self.thumbnailcircuit_ceiling= Image.new('RGB', (self.basewidth,hsize), self.BLACK)

        self.draw = ImageDraw.Draw(self.thumbnailcircuit)
        self.draw_ceiling = ImageDraw.Draw(self.thumbnailcircuit_ceiling)

        #sizes of circuit are in mm.
        self.width,self.height = 22000,8000
        self.ceiling_altitude = 2200
        self.draw_thumbnail_floor()
        self.draw_thumbnail_ceiling()

        self.compile()  
#        self.display_thumbnailcircuit()

        self.thumbnailcircuit.save(self.track_folder+"/gen_track.jpg")
        self.thumbnailcircuit_ceiling.save(self.track_folder+"/gen_ceiling.jpg")

# prepare the temporary image that will contain a circle of 360 degrees around the car

        #self.image_360 = Image.new('RGB', (2*self.visible_surrounding,2*self.visible_surrounding), self.BLACK)
        #self.image_360_ceiling = Image.new('RGB', (2*self.visible_surrounding,2*self.visible_surrounding), self.BLACK)
        #put all the visible surroundings into an image : 

        #output images with the reaquired pov angle.


        # define the Images output
        #self.output_image = Image.new('RGB', (160, 120),(0,0,0))

        #self.np_output_img=np.array(self.output_image)

    def randomize_color(self,color,maxf1,maxf2):
        old_color=color       
        n1= random.randint(-maxf1,maxf1)
        n2= random.randint(-maxf1,maxf1)
        n3= random.randint(-maxf1,maxf1)
        n4=random.randint(-maxf2,maxf2) 

        color=(color[0]+n1+n4, color[0]+n2+n4, color[0]+n3+n4 )
        # put old_color to avoid randomization  : 
        return  old_color
        
    def draw_thumbnail_floor(self):
        for i in range(self.Matrix_Floor_width) :
           for j in range(self.Matrix_Floor_height):
              ktexindex,ktex=self.Matrix_Floor[j][i]
              im2=self.txlist[ktexindex][ktex]
              self.thumbnailcircuit.paste(im2, (int(i*500/self.scalefactor), int(j*500/self.scalefactor)))

    def draw_thumbnail_ceiling(self):
        for i in range(self.Matrix_Ceiling_width) :
           for j in range(self.Matrix_Ceiling_height):
              ktexindex,ktex=self.Matrix_Ceiling[j][i]
              im2=self.txlist[ktexindex][ktex]
              self.thumbnailcircuit_ceiling.paste(im2, (int(i*500/self.scalefactor), int(j*500/self.scalefactor)))

    def translate_circuit(self,X0,Y0):
        for i in range(0,len(self.defc)) :
            self.defc[i][0] += X0
            self.defc[i][1] += Y0

    def display_thumbnailcircuit(self):
        RGB_img = cv2.cvtColor(np.array(self.thumbnailcircuit), cv2.COLOR_BGR2RGB)
        cv2_imshow(RGB_img)

        RGB_img = cv2.cvtColor(np.array(self.thumbnailcircuit_ceiling), cv2.COLOR_BGR2RGB)
        cv2_imshow(RGB_img)

    def get_file_by_prefix(self,path,prefix):
        result=""
        for i in os.listdir(path):
            if os.path.isfile(os.path.join(path,i)) and i.startswith(prefix) :
                result=i
        return result

    def Add_texture(self,ktindex,size):
        #size  : size of texture bitmap in our circuit image ( already converted with the reducefactor factor): 
        n_instance=0
        instance_exists = True
        # Add the container for all the instances of the texture :
        self.txlist.append([])
        self.txnumbers.append([])

        # now populate it with the images of each texture instance :
        while instance_exists :
            prefix=str(ktindex)+"_"+str(n_instance)+"_"
            filename=self.get_file_by_prefix(self.texture_folder,prefix)
            if filename !=  "" :
                img=Image.open(self.texture_folder+"/"+filename).convert("RGB")
                img=img.resize(size,Image.ANTIALIAS)
                self.txlist[ktindex].append(img)
                n_instance+=1
                self.txnumbers[ktindex]=n_instance
            else : 
                instance_exists=False 
        return n_instance

    def load_textures(self) :
        self.tile_w, self.tile_h = self.Matrix_Floor_tile_width,self.Matrix_Floor_tile_height
        size=(int(self.tile_w/self.scalefactor), int(self.tile_h/self.scalefactor))
        ktindex=0
        texture_file_exists = True

        while texture_file_exists :
            nb_instances=self.Add_texture(ktindex,size)
            if nb_instances == 0 : 
            	texture_file_exists=False
            else : 
                ktindex+=1
        print ( "Nb of textures loaded : " + str(ktindex))
        return ktindex





        # Sides

        # textures needs to be 1000=1m  width for the sides and 2000 width for the long side.
        #self.side1=np.array(Image.open(self.texture_folder+"/side1.png").convert("RGB"))
        #self.side1_w, self.side1_h = self.side1.shape[1],self.side1.shape[0]
        #self.side2=np.array(Image.open(self.texture_folder+"/side2.png").convert("RGB"))
        #self.side2_w, self.side2_h = self.side2.shape[1],self.side2.shape[0]
        #self.side3=np.array(Image.open(self.texture_folder+"/side3.png").convert("RGB"))
        #self.side3_w, self.side3_h = self.side3.shape[1],self.side3.shape[0]
        #self.side4=np.array(Image.open(self.texture_folder+"/side4.png").convert("RGB"))
        #self.side4_w, self.side4_h =self.side4.shape[1],self.side4.shape[0]

    def init_empty_map(self) :

        # initiate a default map matrix  in case none is given in the folder : 
        # default texture wil be 0

        # to be addeed : check if there is no matrix floor, then if not : generate it :
        for i in range(0, self.Matrix_Floor_width, 1):
            for j in range(0, self.Matrix_Floor_height, 1):
                if j>=self.Matrix_Floor_height-4 : 
                   ktex_index=1
                else : 
                   ktex_index=0  # first texture

                   if i %3 == 0 and j %3 == 1 : 
                      ktex_index=1

                self.Simple_Matrix_Floor[int(j)][int(i)]=ktex_index
                ktex_index=2
                if i %3 == 0 and j %3 == 1 : 
                   ktex_index=4
                else:
                   rnd=random.randint(0,30)
                   if rnd %10 == 0 : ktex_index =5
                   if rnd == 5 : ktex_index =6

                self.Simple_Matrix_Ceiling[int(j)][int(i)]=ktex_index



        # generate a texture_map json file if it does not exists:
        #try:

        test=json.dumps(self.Simple_Matrix_Floor)
        test = test.replace("],", "],\n")
        with open(self.track_folder+"/floor_texture_map.json", 'w') as fp:
                fp.write(test)
               # json.dump(self.Simple_Matrix_Floor,fp)

        test=json.dumps(self.Simple_Matrix_Ceiling)
        test = test.replace("],", "],\n")
        with open(self.track_folder+"/ceiling_texture_map.json", 'w') as fp:
                fp.write(test)
              #  json.dump(self.Simple_Matrix_Ceiling,fp)

    def load_map(self):

        # Load simple matrix from file
        try:
            with open(self.track_folder+"/floor_texture_map.json", 'r') as f:
                self.Simple_Matrix_Floor = json.load(f)
        except FileNotFoundError:
                print("Error loading floor_texture_map.json")

        try:
            with open(self.track_folder+"/ceiling_texture_map.json", 'r') as f:
                self.Simple_Matrix_Ceiling = json.load(f)
        except FileNotFoundError:
                print("Error loading ceiling_texture_map.json")


        #randomize instances of textures : 
        for i in range(0, self.Matrix_Floor_width, 1):
            for j in range(0, self.Matrix_Floor_height, 1):
                   ktex_index=self.Simple_Matrix_Floor[j][i]
                   nb_texture_instance=self.txnumbers[ktex_index]
                   ktex= random.randint(0,nb_texture_instance-1)
                   self.Matrix_Floor[int(j)][int(i)]=[ktex_index,ktex]

        for i in range(0, self.Matrix_Ceiling_width, 1):
            for j in range(0, self.Matrix_Ceiling_height, 1):
                   ktex_index=self.Simple_Matrix_Ceiling[j][i]
                   nb_texture_instance=self.txnumbers[ktex_index]
                   ktex= random.randint(0,nb_texture_instance-1)
                   self.Matrix_Ceiling[int(j)][int(i)]=[ktex_index,ktex]



    def compile(self):
        cwidth=self.circuit_info["LINE_WIDTH"]  # 5cm
        p00 = self.defc[0]

        first=True
        for p in self.defc :
          if first == False :
             self.draw_portion(p0,p,self.WHITE_LEFT,self.WHITE_CENTER,self.WHITE_RIGHT,cwidth)
             p0=p
          else : 
             p0=p
             first=False
        
        self.draw_portion(p0,p00,self.WHITE_LEFT,self.WHITE_CENTER,self.WHITE_RIGHT,cwidth)

    def draw_portion(self,p1,p2,color_left,color_center,color_right,cwidth):

 
        if (p1[0] == p2[0]) and (p1[1] == p2[1]) : 
           self.draw_curve(p1,p2,color_left,color_center,color_right,cwidth)
        else :
           self.draw_line(p1,p2,color_left,color_center,color_right,cwidth)


    def draw_curve(self,p1,p2,color_left,color_center,color_right,cwidth):
        cx1,cy1,ra1x,ra1y,rb1x,rb1y=self.get_points(p1)
        cx2,cy2,ra2x,ra2y,rb2x,rb2y=self.get_points(p2)

        alpha1=p1[2]
        r1=p1[3]

        alpha2=p2[2]
        r2=p1[4]

        self.arc(cx1,cy1,alpha1,alpha2,r1,color_left,cwidth)
        self.arc(cx1,cy1,alpha1,alpha2,r2,color_right,cwidth)


    def arc(self,cx,cy,anglestart,anglestop,r,color,cwidth,vertices=30):
        dteta=(anglestop-anglestart)/vertices
        x0=cx+r*math.cos(anglestart+0)
        y0=cy+r*math.sin(anglestart+0)
        for n in range(0,vertices+1):
            x1=cx+r*math.cos(anglestart+n*dteta)
            y1=cy+r*math.sin(anglestart+n*dteta)

            color=self.randomize_color(color,2,10)
            self.draw.line( ( x0*self.wpercent,y0*self.wpercent,x1*self.wpercent,y1*self.wpercent),
                fill=color, width=round(cwidth*self.wpercent))
            x0=x1
            y0=y1


    def draw_line(self,p1,p2,color_left,color_center,color_right,cwidth):
        cx1,cy1,ra1x,ra1y,rb1x,rb1y=self.get_points(p1)
        cx2,xy2,ra2x,ra2y,rb2x,rb2y=self.get_points(p2)

        self.draw.line( (round(ra1x*self.wpercent), round(ra1y*self.wpercent),
            round(ra2x*self.wpercent),round( ra2y*self.wpercent)),
            fill=self.randomize_color(color_left,2,10), width=round(cwidth*self.wpercent))

        self.draw.line(( round(rb1x*self.wpercent), round(rb1y*self.wpercent),
            round(rb2x*self.wpercent), round(rb2y*self.wpercent)),
            fill=self.randomize_color(color_right,2,10),width=round(cwidth*self.wpercent)            )

    def get_points(self,p):
        cx,cy,alpha,ra,rb= p[0],p[1],p[2],p[3],p[4]
        cy=self.height-cy
        #alpha=alpha+pi/2
        rax=cx+ra*math.cos(alpha)
        ray=cy+ra*math.sin(alpha)
        rbx=cx+rb*math.cos(alpha)
        rby=cy+rb*math.sin(alpha)
        return cx,cy,rax,ray,rbx,rby


    def get_angle_index(self,angle):
        r=math.fmod(angle,2*pi)

        if r<0 : 
          r=r+2*pi

        n_angle= int(r*360/(2*pi))
        if n_angle>=360 : n_angle -=360

        return n_angle





# STORM : 
H=230
H=830

#Y0=1000 #mm
#40px=46cm
Y0=1560 #mm
D0=650 # mm
L=250

#Car Geometry : 
visibility=3000
RAD_MAX_WHEEL=pi/4
CAR_WIDTH=150

LOW_D=100
LOW_D_pixel=70


circuit1=circuit(track_folder="Tracks/Cristallin",ClearFolder=False,scalefactor=4)
circuit2=circuit(track_folder="Tracks/Epita",ClearFolder=False,scalefactor=4)
# scalefactor=4 : 1 pixel = 4mm
