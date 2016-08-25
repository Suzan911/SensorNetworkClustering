from matplotlib import pyplot as plt
import numpy as np


def inch2figu(pos,fsize):
    '''
    Convert a four-el position vector to units normalized by figure
    size. Note that position and figure size units must the same.
    
    Args:
        pos (4-el array): xoffset, yoffset, width, height [phys units]
        fsize (2-el array): figure width and height [phys units].
    
    Returns:
        4-el array: normalized position array
    '''

    pos = np.array(pos,dtype='float')

    pos[0] = pos[0]/fsize[0]
    pos[1] = pos[1]/fsize[1]
    pos[2] = pos[2]/fsize[0]
    pos[3] = pos[3]/fsize[1]

    return pos

    
class myfig:
    '''
    This class stores information about a figure with multiple axes 
    aligned vertically.
    '''
    
    def __init__(self, **kwargs):
        
        
        self.subploth   = np.array(kwargs.get('subploth',([5.,5.])),dtype='float')
        self.subplotw   = kwargs.get('subplotw',5.)
        self.x0         = kwargs.get('x0',.5)
        self.y0         = kwargs.get('y0',.5)
        self.yspace     = kwargs.get('yspace',.5)
        self.x1         = kwargs.get('x1',.25)
        self.y1         = kwargs.get('y1',.25)
        
        
        self.fheight = (self.x0
            +np.sum(self.subploth)
            +self.yspace*(self.nsubpl()-1)
            +self.x1)
            
        self.fwidth = (self.y0
            +self.subplotw
            +self.y1)
            
        F = plt.gcf()
        F.set_size_inches([self.fwidth,self.fheight])
        
        self.F = F
        
    
    def nsubpl(self):
        return self.subploth.size
    
    
        
    def makeAxesPos(self,pos):
        pos = inch2figu(pos,[self.fwidth,self.fheight])
        
        # Make object's figure current figure
        pass
        
        ax = plt.axes(pos)
        return ax

    
    
    def makeAxes(self,i):
        '''
        Create an axes object in the objects figure
        Args:
            i (int): Index of plot, in the range 0..nsubpl-1. Zero is 
            the bottom axes.
            
        Returns:
            axes object: 
        '''
        
        # Compute position vector for i-th figure
        x0 = self.x0
        y0 = (self.y0 
            + np.sum(self.subploth[:i])
            + i*self.yspace)
        
        w = self.subplotw
        h = self.subploth[i]
        
        ax = self.makeAxesPos([x0,y0,w,h])
        
        return ax
        