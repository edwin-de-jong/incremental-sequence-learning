import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops

class Stats():
    def __init__( self, args, nrseq, label ):
        self.nrseq = nrseq
        self.label = label
        self.stats_stroke = RMSEStat( args, nrseq, label + 'stroke' )      
        self.stats_correct = AvgStat( args, nrseq, label + 'correct' )      
        self.stats_correctfrac = AvgStat( args, nrseq, label + 'correctfrac' )      

    def reset( self ):                
        self.stats_stroke.reset()
        self.stats_correct.reset()
        self.stats_correctfrac.reset()
    
class RMSEStat():
    def __init__( self, args, nrseq, label ):
        self.args = args
        self.nrseq = nrseq
        self.label = label
        self.reset()
        
    def log_sse_sequential( self, sse, nrpoints ): #no index given --> cycle through all sequences
        self.log_sse( self.pointer, sse, nrpoints)            
        self.pointer += 1
        if ( self.pointer >= self.nrseq):
            self.pointer = 0

    def log_sse( self, sequence_index, sse, nrpoints ): #replace previous value of this specific example, so that stats always reflect all examples
        curval = self.sse[ sequence_index ]
        diff = sse - curval
        self.sse[ sequence_index ] = sse
        self.sse_sum += diff
        
        curval = self.nrpoints[ sequence_index ]
        diff = nrpoints - curval
        self.nrpoints[ sequence_index ] = nrpoints
        self.totnrpoints += diff
            
    def reset( self ):                
        self.sse = np.zeros( (self.nrseq), dtype=np.float32 )
        self.sse_sum = 0

        self.nrpoints = np.zeros( (self.nrseq), dtype=np.float32 )
        self.totnrpoints = 0
        
        self.pointer = 0
        
    def rmse( self ):
        return np.sqrt( self.sse_sum / max( 1, self.totnrpoints ) )
            
class AvgStat():
    def __init__( self, args, nrseq, label ):
        self.nrseq = nrseq
        self.label = label
        self.reset()
        
    def log_value_sequential( self, value, nrpoints ):
        self.log_value( self.pointer, value, nrpoints)
        self.pointer += 1
        if ( self.pointer >= self.nrseq):
            self.pointer = 0

    def log_value( self, sequence_index, value, nrpoints ):
        curval = self.values[ sequence_index ]
        diff = value - curval
        self.values[ sequence_index ] = value
        self.values_sum += diff
        
        curval = self.nrpoints[ sequence_index ]
        diff = nrpoints - curval
        self.nrpoints[ sequence_index ] = nrpoints
        self.totnrpoints += diff
            
    def reset( self ):
        self.nrpoints = 0
                
        self.values = np.zeros( (self.nrseq), dtype=np.float32 )
        self.values_sum = 0

        self.nrpoints = np.zeros( (self.nrseq), dtype=np.float32 )
        self.totnrpoints = 0
        
        self.pointer = 0
        
    def average( self ):
        return self.values_sum / max( 1, self.totnrpoints ) 
