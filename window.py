import cv2
import numpy as np

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import sys
from pathfile import *
from config import *
#helper modules
import glutil
from vector import Vec
from datetime import datetime



class window():
    def __init__(self, X, I):
        self.X = X
        self.I = I
        self.shape = shape
        self.projection = projection
        self.Nx = len(self.X)
        self.img_dir = img_dir
        self.alpha = 1.
     
        #mouse handling for transforming scene
        self.mouse_down = False
        self.mouse_old = Vec([0., 0.])
        self.rotate = Vec([0., 0., 180.])
        self.translate = Vec([0., 0., 0.])
        self.initrans = Vec([0., 0., -self.shape[2]*2])

        self.width = 800
        self.height = 800
        
        self.capture = capture
        self.video_dir = video_dir
        if self.capture:
            self.captureInit()
        
        glutInit(sys.argv)
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
        glutInitWindowSize(self.width, self.height)
        glutInitWindowPosition(0, 0)
        self.win = glutCreateWindow("")

        #gets called by GLUT every frame
        glutDisplayFunc(self.draw)

        #handle user input
        glutKeyboardFunc(self.on_key)
        glutMouseFunc(self.on_click)
        glutMotionFunc(self.on_mouse_motion)
        
        #this will call draw every 30 ms
        glutTimerFunc(30, self.timer, 30)

        #setup OpenGL scene
        self.glinit()
        self.loadVBO()
        glutMainLoop()
        

    
    def glinit(self):
        glViewport(0, 0, self.width, self.height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        if (self.projection == 'perspective'):
            gluPerspective(60., self.width / float(self.height), .1, self.shape[2]*4)
        if (self.projection == 'ortho'):
            glOrtho(-self.shape[0]/2,self.shape[0]/2,-self.shape[1]/2,self.shape[1]/2, .1, self.shape[2]*4)
        glMatrixMode(GL_MODELVIEW)
        glLineWidth(2.)
        glEnable (GL_LINE_SMOOTH)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)
    
    def generate_fname(self, ext):
        systime = datetime.now()
        hms = str(systime.hour).zfill(2)+str(systime.minute).zfill(2)+str(systime.second).zfill(2)
        return hms+ext
        
    def captureInit(self):
        self.fname = self.video_dir+'/'+self.generate_fname('.avi')
        self.video = cv2.VideoWriter(self.fname,cv2.VideoWriter_fourcc('X','2','6','4'),30,(self.width,self.height))
        self.pbyData = np.zeros((self.width,self.height,3), dtype = np.ubyte)
    
    ###GL CALLBACKS
    def timer(self, t):
        glutTimerFunc(t, self.timer, t)
        glutPostRedisplay()

    def on_key(self, *args):
        ESCAPE = '\033'
        if args[0] == ESCAPE or args[0] == 'q':
            self.capture = False
            self.video.release()
            sys.exit(0)
            
        if args[0] == 's':
            if self.capture:
                print 'Stopping video capture'
                self.capture = False
                self.video.release()
            else:
                self.captureInit()
                print 'Starting video capture. Saving to '+self.fname
                self.capture = True               

        if args[0] == 'p':
            self.scrnData = np.zeros((self.width,self.height,3), dtype = np.ubyte)
            glReadPixels(0, 0, self.width, self.height, GL_BGR, GL_UNSIGNED_BYTE, self.scrnData)
            scipy.misc.imsave(self.img_dir+'/'+self.generate_fname('.png'), np.flipud(self.scrnData))
            
            
    def on_click(self, button, state, x, y):
        if state == GLUT_DOWN:
            self.mouse_down = True
            self.button = button
        else:
            self.mouse_down = False
        self.mouse_old.x = x
        self.mouse_old.y = y

    
    def on_mouse_motion(self, x, y):
        dx = x - self.mouse_old.x
        dy = y - self.mouse_old.y
        if self.mouse_down and self.button == 0: #left button
            self.rotate.x += dy * .2
            self.rotate.y += dx * .2
        elif self.mouse_down and self.button == 2: #right button
            if self.projection == 'perspective':
                self.translate.z -= dy * .1
            if self.projection == 'ortho':
                self.alpha -= dy*2e-2
        self.mouse_old.x = x
        self.mouse_old.y = y
    ###END GL CALLBACKS    
    
    def render(self):

        for i in range(0,self.Nx):
            color = self.Col_vbo[i].data
            color[:,3] = self.I[i]*self.alpha
            self.Col_vbo[i].set_array(color)
            self.Col_vbo[i].bind()
            glColorPointer(4, GL_FLOAT, 0, None)
            self.Pos_vbo[i].bind()
            glVertexPointer(4, GL_FLOAT, 0, None)

            glEnableClientState(GL_VERTEX_ARRAY)
            glEnableClientState(GL_COLOR_ARRAY)

            glDrawArrays(GL_LINE_STRIP, 0, self.X[i].shape[0])

            glDisableClientState(GL_COLOR_ARRAY)
            glDisableClientState(GL_VERTEX_ARRAY)
     
        
    def draw(self):              
        glFlush()

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        #handle mouse transformations
        glTranslatef(self.initrans.x, self.initrans.y, self.initrans.z)
        glTranslatef(self.translate.x, self.translate.y, self.translate.z)
        glRotatef(self.rotate.x, 1, 0, 0)
        glRotatef(self.rotate.y, 0, 1, 0)
        glRotatef(self.rotate.z, 0, 0, 1)
        
        self.render()
        
        if self.capture:
            glReadPixels(0, 0, self.width, self.height, GL_BGR, GL_UNSIGNED_BYTE, self.pbyData)
            self.video.write(self.pbyData)

        glutSwapBuffers()
        
        
        
    def loadVBO(self):    
        from OpenGL.arrays import vbo
        
        self.Pos_vbo = []
        self.Col_vbo = []
        
        for i in range(0,self.Nx):
            pos_vbo = vbo.VBO(data=self.X[i], usage=GL_DYNAMIC_DRAW, target=GL_ARRAY_BUFFER)
            pos_vbo.bind()
            self.Pos_vbo = self.Pos_vbo + [pos_vbo]    
            color = np.zeros_like(self.X[i])
            color[:,0] = 1.
            color[:,1] = 1.
            color[:,2] = 1.
            color[:,3] = self.I[i]*self.alpha
            col_vbo = vbo.VBO(data=color, usage=GL_DYNAMIC_DRAW, target=GL_ARRAY_BUFFER)
            col_vbo.bind()
            self.Col_vbo = self.Col_vbo + [col_vbo]   
        return self
