import os
import sys
import pickle
import numpy as np
import xml.etree.ElementTree as ET
import random
import svgwrite
from IPython.display import SVG, display
import tensorflow as tf


def get_bounds( data, factor ):
  min_x = 0
  max_x = 0
  min_y = 0
  max_y = 0
    
  abs_x = 0
  abs_y = 0
  for i in range( len( data ) ):
    x = float( data[ i, 0 ] )/factor
    y = float( data[ i, 1 ] )/factor
    abs_x += x
    abs_y += y
    min_x = min( min_x, abs_x )
    min_y = min( min_y, abs_y )
    max_x = max( max_x, abs_x )
    max_y = max( max_y, abs_y )
    
  return ( min_x, max_x, min_y, max_y )

# version where each path is entire stroke ( smaller svg size, but have to keep same color )
def draw_strokes( data, factor = 10, svg_filename = 'sample.svg' ):
  min_x, max_x, min_y, max_y = get_bounds( data, factor )
  dims = ( 50 + max_x - min_x, 50 + max_y - min_y )
    
  dwg = svgwrite.Drawing( svg_filename, size = dims )
  dwg.add( dwg.rect( insert = ( 0, 0 ), size = dims, fill = 'white' ) )

  lift_pen = 1
    
  abs_x = 25 - min_x 
  abs_y = 25 - min_y
  p = "M%s, %s " % ( abs_x, abs_y )
    
  command = "m"

  for i in range( len( data ) ):
    if ( lift_pen == 1 ):
      command = "m"
    elif ( command != "l" ):
      command = "l"
    else:
      command = ""
    x = float( data[ i, 0 ] )/factor
    y = float( data[ i, 1 ] )/factor
    lift_pen = data[ i, 2 ]
    p += command+str( x )+", "+str( y )+" "

  the_color = "black"
  stroke_width = 1

  dwg.add( dwg.path( p ).stroke( the_color, stroke_width ).fill( "none" ) )

  dwg.save( )
  display( SVG( dwg.tostring( ) ) )

def draw_strokes_eos_weighted( stroke, param, factor = 10, svg_filename = 'sample[ A_eos.svg' ):
  c_data_eos = np.zeros( ( len( stroke ), 3 ) )
  for i in range( len( param ) ):
    c_data_eos[ i, : ] = ( 1-param[ i ][ 6 ][ 0 ] )*225 # make color gray scale, darker = more likely to eos
  draw_strokes_custom_color( stroke, factor = factor, svg_filename = svg_filename, color_data = c_data_eos, stroke_width = 3 )

def draw_strokes_random_color( stroke, factor = 10, svg_filename = 'sample_random_color.svg', per_stroke_mode = True ):
  c_data = np.array( np.random.rand( len( stroke ), 3 )*240, dtype = np.uint8 )
  if per_stroke_mode:
    switch_color = False
    for i in range( len( stroke ) ):
      if switch_color == False and i > 0:
        c_data[ i ] = c_data[ i-1 ]
      if stroke[ i, 2 ] < 1: # same strike
        switch_color = False
      else:
        switch_color = True
  draw_strokes_custom_color( stroke, factor = factor, svg_filename = svg_filename, color_data = c_data, stroke_width = 2 )

def draw_strokes_custom_color( data, factor = 10, svg_filename = 'test.svg', color_data = None, stroke_width = 1 ):
  min_x, max_x, min_y, max_y = get_bounds( data, factor )
  dims = ( 50 + max_x - min_x, 50 + max_y - min_y )
    
  dwg = svgwrite.Drawing( svg_filename, size = dims )
  dwg.add( dwg.rect( insert = ( 0, 0 ), size = dims, fill = 'white' ) )

  lift_pen = 1
  abs_x = 25 - min_x 
  abs_y = 25 - min_y

  for i in range( len( data ) ):

    x = float( data[ i, 0 ] )/factor
    y = float( data[ i, 1 ] )/factor

    prev_x = abs_x
    prev_y = abs_y

    abs_x += x
    abs_y += y

    if ( lift_pen == 1 ):
      p = "M "+str( abs_x )+", "+str( abs_y )+" "
    else:
      p = "M +"+str( prev_x )+", "+str( prev_y )+" L "+str( abs_x )+", "+str( abs_y )+" "

    lift_pen = data[ i, 2 ]

    the_color = "black"

    if ( color_data is not None ):
      the_color = "rgb( "+str( int( color_data[ i, 0 ] ) )+", "+str( int( color_data[ i, 1 ] ) )+", "+str( int( color_data[ i, 2 ] ) )+" )"

    dwg.add( dwg.path( p ).stroke( the_color, stroke_width ).fill( the_color ) )
  dwg.save( )
  display( SVG( dwg.tostring( ) ) )

def draw_strokes_pdf( data, param, factor = 10, svg_filename = 'sample_pdf.svg' ):
  min_x, max_x, min_y, max_y = get_bounds( data, factor )
  dims = ( 50 + max_x - min_x, 50 + max_y - min_y )

  dwg = svgwrite.Drawing( svg_filename, size = dims )
  dwg.add( dwg.rect( insert = ( 0, 0 ), size = dims, fill = 'white' ) )

  abs_x = 25 - min_x 
  abs_y = 25 - min_y

  num_mixture = len( param[ 0 ][ 0 ] )

  for i in range( len( data ) ):

    x = float( data[ i, 0 ] )/factor
    y = float( data[ i, 1 ] )/factor

    for k in range( num_mixture ):
      pi = param[ i ][ 0 ][ k ]
      if pi > 0.01: # optimisation, ignore pi's less than 1% chance
        mu1 = param[ i ][ 1 ][ k ]
        mu2 = param[ i ][ 2 ][ k ]
        s1 = param[ i ][ 3 ][ k ]
        s2 = param[ i ][ 4 ][ k ]
        sigma = np.sqrt( s1*s2 )
        dwg.add( dwg.circle( center = ( abs_x+mu1*factor, abs_y+mu2*factor ), r = int( sigma*factor ) ).fill( 'red', opacity = pi/( sigma*sigma*factor ) ) )

    prev_x = abs_x
    prev_y = abs_y

    abs_x += x
    abs_y += y


  dwg.save( )
  display( SVG( dwg.tostring( ) ) )
  
class DataLoader( ):

  def getRandValue( self ):
    value = self.randvalues[ self.randvaluepointer ]
    self.randvaluepointer += 1
    if ( self.randvaluepointer >= self.nrrandvalues ):
        self.randvaluepointer = 0
    return value
        
  def createRandValues( self ):
    self.nrrandvalues = 1000
    self.randvalues = np.zeros( ( self.nrrandvalues ), dtype = np.float32 )
    for i in range( self.nrrandvalues ):
        value = random.random( )
        self.randvalues[ i ] = value
    self.randvaluepointer = 0

  def getClassLabels( self ):
    if self.train:
      fn = self.data_dir + "trainlabels.txt"
    else:
      fn = self.data_dir + "testlabels.txt"
    classlabels = np.loadtxt( fn )
    classlabels = classlabels[ :self.nrinputfiles ]
    return classlabels

  def findAvailableExamples( self, args ):
    self.availableExamples = [ ]
    findexamples = True
    if findexamples:
      for i in range( len( self.classlabels ) ):
        if ( self.classlabels[ i ] < args.curnrdigits ):
          self.availableExamples.append( i )
    self.availableExamples = np.array( self.availableExamples )
  
  def __init__( self, datadir, args, totnrfiles, curnrexamples, seqlength = 0, train = 1, file_label = "", print_input = 0, rangemin = 0, rangelen = 0 ):

    random.seed( 100*args.runnr )
    np.random.seed( 100*args.runnr )
    tf.set_random_seed( 100*args.runnr )    
    self.args = args

    self.data_dir = datadir
    self.train = train
    if self.train:
      self.traintest = "train"
    else:
      self.traintest = "test"
    self.rangemin = rangemin
    self.rangelen = rangelen
    self.nrinputfiles = totnrfiles

    self.curnrexamples = curnrexamples
    self.nrseq_per_batch = args.nrseq_per_batch
    self.file_label = file_label
    self.print_input = print_input
    self.nrinputvars_data = self.getInputVectorLength( args )    
    self.max_seq_length = args.max_seq_length
    
    self.nrsequenceinputs = 4 #dx dy eos eod
    self.nrauxinputvars = args.nrClassOutputVars #either [ 0..9 dx dy eos eod ] or [ dx dy eos ]

    strokedatafile = os.path.join( self.data_dir, "strokes_"+self.traintest+"ing_data"+ file_label+args.explabel+ ".cpkl" )
    raw_data_dir = self.data_dir+"/lineStrokes"

    print ( "creating data cpkl file from source data" )
    self.preprocess( args, raw_data_dir, strokedatafile )

    if ( seqlength > 0 ): #provided
      self.seq_length = seqlength
    else:
      self.seq_length = min( self.max_seq_length, args.maxdigitlength_nrpoints )

    self.load_preprocessed( args, strokedatafile )

    self.classlabels = self.getClassLabels( )    
    self.findAvailableExamples( args )

    self.nrbatches_per_epoch = max( 1, int( self.curnrexamples / self.nrseq_per_batch ) )
    print ( "curnrexamples", self.curnrexamples, "seq_length", self.seq_length, " --> nrbatches_per_epoch: ", self.nrbatches_per_epoch )

    print ( "loaded data" )
    self.reset_batch_pointer( args )

  def constructInputFileName( self, args, file_label, imgnr ):
    filename = self.data_dir + self.traintest + 'img' + file_label + '-' + str( imgnr ) + '-targetdata.txt' #currently, we expect 14 inputs
    return filename

  def getInputVectorLength( self, args ):
    result = [ ]

    filename = self.constructInputFileName( args, self.file_label, imgnr = 0 )

    with open( filename ) as f:
      points = [ ]
      line = f.readline( )
      print ( "read sample line from inputdata file: ", line )
      nrs = [ float( x ) for x in line.split( ) ]
      length = len( nrs )
      print ( "Determined nrinputvars based on data: ", length )
    self.nrinputvars_data = length
    return length

  def preprocess( self, args, data_dir, strokedatafile ):
    filelist = [ ]

    if len( args.fileselection )>0:
      fileselection = ' '.join( args.fileselection )
      if len( fileselection )>0:
        fileselection = [ int( s ) for s in fileselection.split( ', ' ) ]

    for imgnr in range( 0, self.nrinputfiles ):
        if len( args.fileselection )>0:
          fname = self.constructInputFileName( args, self.file_label, fileselection[ imgnr ] )
        else:
          fname = self.constructInputFileName( args, self.file_label, imgnr )
        filelist.append( fname )

    def getStrokes( filename, nrauxinputvars ): #returns array of arrays with points
      result_points = [ ]
      result_auxinputs = [ ]
      nrsequencevars = 4
      dxmin = 1e100
      dxmax = -1e100
      dymin = 1e100
      dymax = -1e100
      nrauxinputs_data = 10

      with open( filename ) as f:
        points = [ ]
        auxinputs = [ ]
        for line in f: # read rest of lines
          nrs = [ float( x ) for x in line.split( ) ]
          auxinputvalues = nrs[ 0:nrauxinputvars ]
          point = nrs[ nrauxinputs_data:nrauxinputs_data+nrsequencevars ] #currently: x, y, end-of-stroke
          points.append( point )
          auxinputs.append( auxinputvalues )
        result_points.append( points )
        result_auxinputs.append( auxinputs )
        pointarray = np.array( points )
        digitlength_nrpoints = len( points )
        dxmin = pointarray[ :, 0 ].min( )
        dxmax = pointarray[ :, 0 ].max( )
        dymin = pointarray[ :, 1 ].min( )
        dymax = pointarray[ :, 1 ].max( )
        ranges = [ dxmin, dxmax, dymin, dymax ]
      return result_auxinputs, result_points, ranges, digitlength_nrpoints

    # converts a list of arrays into a 2d numpy int16 array
    def convert_stroke_to_array( stroke ):

      n_point = 0
      for i in range( len( stroke ) ):
        n_point += len( stroke[ i ] )

      prev_x = 0
      prev_y = 0
      counter = 0
      nrsequencevars = 4
      stroke_data = np.zeros( ( n_point, nrsequencevars ), dtype = np.int16 )

      for j in range( len( stroke ) ):
        for k in range( len( stroke[ j ] ) ):
          for s in range( nrsequencevars ):
            stroke_data[ counter, s ] = int( stroke[ j ][ k ][ s ] ) 
          counter += 1
      return stroke_data

    # converts a list of arrays into a 2d numpy int16 array
    def convert_auxinputs_to_array( auxinputs, nrauxinputvars ):

      n_point = 0
      for i in range( len( auxinputs ) ):
        n_point += len( auxinputs[ i ] )
      auxinputdata = np.zeros( ( n_point, nrauxinputvars ), dtype = np.int16 )

      prev_x = 0
      prev_y = 0
      counter = 0

      for j in range( len( auxinputs ) ):
        for k in range( len( auxinputs[ j ] ) ):
          for a in range( nrauxinputvars ):
            auxinputdata[ counter, a ] = int( auxinputs[ j ][ k ][ a ] ) 
          counter += 1
      return auxinputdata

    # preprocess body: build stroke array
    strokearray = [ ]
    auxinputarray = [ ]
    rangelist = [ ]
    self.seqlengthlist = [ ] 
    if self.train:
      args.maxdigitlength_nrpoints = 0
    digitlengthsum = 0
    for i in range( len( filelist ) ):
      print ( 'dataloader', self.traintest, 'processing '+filelist[ i ] )
      [ auxinputs, strokeinputs, ranges, digitlength_nrpoints ] = getStrokes( filelist[ i ], self.nrauxinputvars )
      strokearray.append( convert_stroke_to_array( strokeinputs ) )
      auxinputarray.append( convert_auxinputs_to_array( auxinputs, self.nrauxinputvars ) )
      rangelist.append( ranges )
      self.seqlengthlist.append( digitlength_nrpoints )
      if self.train:
        args.maxdigitlength_nrpoints = max( args.maxdigitlength_nrpoints, digitlength_nrpoints )
        digitlengthsum += digitlength_nrpoints
        
    rangearray = np.array( rangelist )
    ranges = [ rangearray[ :, 0 ].min( ), rangearray[ :, 1 ].max( ), rangearray[ :, 2 ].min( ), rangearray[ :, 3 ].max( ) ]
    print ( "found overall ranges", ranges )
    self.avgseqlength = digitlengthsum / len( filelist )
    print( "dataloader: found avg seq length: ", self.avgseqlength )
    print ( "found maxdigitlength_nrpoints", args.maxdigitlength_nrpoints )

    f = open( strokedatafile, "wb" )
    pickle.dump( strokearray, f )
    pickle.dump( auxinputarray, f )
    pickle.dump( ranges, f )
    pickle.dump( self.seqlengthlist, f )
    f.close( )

  def load_preprocessed( self, args, strokedatafile ):
    f = open( strokedatafile, "rb" )
    self.strokedataraw = pickle.load( f )
    self.auxdataraw = pickle.load( f )
    self.ranges = pickle.load( f )
    self.seqlengthlist = pickle.load( f )
    f.close( )
      
    print ( "loaded ranges", self.ranges )
    print ( "rangemin", self.rangemin, "rangelen", self.rangelen )

    self.strokedata = [ ] #contains one array per file
    self.auxdata = [ ]
    counter = 0

    for data_el in self.strokedataraw:
        data = np.array( np.zeros( ( self.seq_length, self.nrsequenceinputs ), dtype = np.float32 ) )
        len_data = len( data )
        nrpoints = min( self.seq_length, len( data_el ) )
        data[ :nrpoints, ] = data_el[ :nrpoints ]
        if ( len( data_el ) > self.seq_length ) and ( self.seq_length >= args.max_seq_length ): 
          data[ self.seq_length-1, 2:4 ] = np.ones( ( 1, 2 ), dtype = np.float32 ) #add eos and eod for sequences exceeding length
        data[ nrpoints:, 0:4 ] = np.zeros( ( len_data - nrpoints, 4 ), dtype = np.float32 ) #pad remainder with zero rows
        data[ :, 0:2 ] -= self.rangemin
        data[ :, 0:2 ] /= self.rangelen
        self.strokedata.append( data )

        counter += 1
    for data_el in self.auxdataraw:
        data = np.array( np.zeros( ( self.seq_length, self.nrauxinputvars ), dtype = np.float32 ) )
        nrpoints = min( self.seq_length, len( data_el ) )
        data[ :nrpoints, ] = data_el[ :nrpoints ]
        data[ nrpoints:self.seq_length, ] = data[ nrpoints-1, ]
        self.auxdata.append( data )
    print ( "#sequences found in data: ", counter )

  def next_batch( self, args, curseqlength ):
      
    # returns a batch of the training data of nrseq_per_batch * seq_length points
    x_batch = [ ]
    y_batch = [ ]
    seqlen = self.seq_length
    sequence_index = [ ]
    use_points_stopcrit = False

    nrpoints_per_batch = 0
    if hasattr( args, 'nrpoints_per_batch' ):
      nrpoints_per_batch = args.nrpoints_per_batch
    if nrpoints_per_batch > 0:
      use_points_stopcrit = True
    batch_nrpoints = 0
    batch_sequencenr = 0
    done = False
    while not done:
      sequence_index.append( self.pointer )      
      strokes = np.copy( self.strokedata[ self.pointer ] )
      auxvalues = np.copy( self.auxdata[ self.pointer ] )

      if args.useStrokeOutputVars:
        ytab = np.copy( np.hstack( [ auxvalues[ 1:seqlen ], strokes[ 1:seqlen ] ] ) )
      else:
        ytab = np.copy( np.hstack( [ auxvalues[ 1:seqlen ] ] ) )
      
      if args.discard_classvar_inputs:
        auxvalues[ : ] = 0
        
      if args.useClassInputVars:
        xtab = np.hstack( [ auxvalues[ :seqlen-1 ], strokes[ :seqlen-1 ] ] )
      else:
        xtab = strokes[ :seqlen-1 ] 

      actual_seq_length = self.seqlengthlist[ self.pointer ]

      firsttrainstep = 0
      if hasattr( args, 'firsttrainstep' ):
        firsttrainstep = args.firsttrainstep
      firsttrainstep = min ( firsttrainstep, actual_seq_length - 1 )
      if firsttrainstep > 0: #remove earlier part from _target_ data, so that it will not be used in loss.
        ytab[ :firsttrainstep, : ] = 0
        
      #only keep points up to current seq_length - 1; e.g. if sequence has 3 points, use 2 pairs of ( x, y ): k = 1.n-1 for x and k = 2..n for y
      firstafter = min( actual_seq_length - 1, curseqlength ) #zero out part after sequence
            
      xtab[ firstafter:, : ] = 0
      ytab[ firstafter:, : ] = 0

      nrusedpoints = firstafter - firsttrainstep
      
      if args.discard_inputs:
        xtab[ : ] = 0

      x_batch.append( np.copy( xtab ) )
      y_batch.append( np.copy( ytab ) )

      self.next_batch_pointer( args )

      batch_sequencenr += 1
      nrseq_per_batch = self.nrseq_per_batch
      if ( not self.train ) and hasattr( args, 'nrseq_per_batch_test' ):
        nrseq_per_batch = args.nrseq_per_batch_test

      if use_points_stopcrit:
        batch_nrpoints += nrusedpoints
        done = batch_nrpoints >= nrpoints_per_batch
      else:
        done = batch_sequencenr >= nrseq_per_batch
  
    return x_batch, y_batch, sequence_index

  
  def selectExamples( self, nrdigits ):
    sample = np.random.permutation( len( self.availableExamples ) )
    return self.availableExamples[ sample ]

  def next_batch_pointer( self, args ):
    self.index += 1
    if ( self.index >= len( self.example_permutation ) ):
      self.reset_batch_pointer( args )
    self.pointer = self.example_permutation[ self.index ]

  def reset_batch_pointer( self, args ):
    self.index = 0

    if ( args.incremental_nr_digits and self.train ):
      self.example_permutation = self.selectExamples( args.curnrdigits )      
    else:
      if self.train:
        self.example_permutation = np.random.permutation( int( self.curnrexamples ) )
      else:
        self.example_permutation = np.arange( 0, int( self.curnrexamples ) )
    self.pointer = self.example_permutation[ self.index ]

