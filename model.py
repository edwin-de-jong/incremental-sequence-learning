import tensorflow as tf
import numpy as np
import random

def get_pi_idx( x, pdf ):
  N = pdf.size
  accumulate = 0
  for i in range( 0, N ):
    accumulate += pdf[ i ]
    if ( accumulate >= x ):
      return i
  print( 'error with sampling ensemble' )
  return -1

class Model( ):

  def get_classvars( self, args, output ):
    z = output    

    last = args.nroutputvars_raw - args.nrClassOutputVars

    classvars = tf.zeros( 1, dtype = tf.float32, name = None )
    classpred = tf.zeros( 1, dtype = tf.float32, name = None )

    if args.nrClassOutputVars > 0:
      classvars = z[ :, last: ]
      classpred = tf.nn.softmax( classvars )

    return [ classvars, classpred ]

    # below is where we need to do MDN splitting of distribution params
  def get_mixture_coef( self, args, output ):
      # returns the tf slices containing mdn dist params
      # ie, eq 18 -> 23 of http://arxiv.org/abs/1308.0850
      z = output    

      #get the remaining parameters
      last = args.nroutputvars_raw - args.nrClassOutputVars
      
      z_eos = z[ :, 0 ]
      z_eos = tf.sigmoid( z_eos ) #eos: sigmoid, eq 18

      z_eod = z[ :, 1 ]
      z_eod = tf.sigmoid( z_eod ) #eod: sigmoid

      z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr = tf.split( z[ :, 2:last ], 6, 1 ) #eq 20: mu1, mu2: no transformation required

      # process output z's into MDN parameters

      # softmax all the pi's:
      max_pi = tf.reduce_max( z_pi, 1, keep_dims = True )
      z_pi = tf.subtract( z_pi, max_pi ) #EdJ: subtract max pi for numerical stabilization

      z_pi = tf.exp( z_pi ) #eq 19
      normalize_pi = tf.reciprocal( tf.reduce_sum( z_pi, 1, keep_dims = True ) )
      z_pi = tf.multiply( normalize_pi, z_pi ) #19

      # exponentiate the sigmas and also make corr between -1 and 1.
      z_sigma1 = tf.exp( z_sigma1 ) #eq 21
      z_sigma2 = tf.exp( z_sigma2 )
      z_corr_tanh = tf.tanh( z_corr ) #eq 22
      z_corr_tanh = .95 * z_corr_tanh #avoid -1 and 1 

      z_corr_tanh_adj = z_corr_tanh 

      return [ z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr_tanh_adj, z_eos, z_eod ]
              
  def sample_gaussian_2d( self, mu1, mu2, s1, s2, rho ):
    mean = [ mu1, mu2 ]
    cov = [ [ s1 * s1, rho * s1 * s2 ], [ rho * s1 * s2, s2 * s2 ] ]
    x = np.random.multivariate_normal( mean, cov, 1 )
    return x[ 0 ][ 0 ], x[ 0 ][ 1 ]

  def tf_2d_normal( self, x1, x2, mu1, mu2, s1, s2, rho ):
      # eq # 24 and 25 of http://arxiv.org/abs/1308.0850
      #dims: mu1, mu2: batch_nrpoints x nrmixtures
      norm1 = tf.subtract( x1, mu1 ) #batch_nrpoints x nrmixtures
      norm2 = tf.subtract( x2, mu2 )
      s1s2 = tf.multiply( s1, s2 )
      normprod = tf.multiply( norm1, norm2 ) #batch_nrpoints x nrmixtures; here x1 and x2 are combined

      epsilon = 1e-10
      self.z = tf.square( tf.div( norm1, s1 + epsilon ) ) + tf.square( tf.div( norm2, s2 + epsilon ) ) - 2 * tf.div( tf.multiply( rho, normprod ), s1s2 + epsilon ) #batch_nrpoints x nrmixtures
      negRho = 1 - tf.square( rho ) #EdJ: Problem: can become 0 if corr is 1 --> denom becomes zero --> nan result, resolved by multiplying z_corr_tanh with 0.95
      result5 = tf.exp( tf.div( - self.z, 2 * negRho ) )
     
      self.denom = 2 * np.pi * tf.multiply( s1s2, tf.sqrt( negRho ) )
      self.result6 = tf.div( result5, self.denom )

      return self.result6 #still batch_nrpoints x nrmixtures

  def getRegularizationTerm( self, args ):

      trainablevars = tf.trainable_variables( )
      self.weights = [ ]
      weightsum = tf.zeros( 1, dtype = tf.float32, name = None )
      nrweights = tf.zeros( 1, dtype = tf.int32, name = None )
      self.maxabsweight = tf.zeros( 1, dtype = tf.float32, name = None )

      for var in trainablevars:
        isBias = var.name.find( "Bias" ) >= 0
        if isBias:
          print ( "Found trainable variable: ", var.name )
        else:
          print ( "Found trainable variable: ", var.name , "; adding to regularization term" )
          self.weights.append( var )
          weightsum = weightsum + tf.reduce_sum( tf.abs( var ) )
          nrweights = tf.add( nrweights , tf.reduce_prod( tf.shape( var ) ) )
          maxval = tf.reduce_max( tf.abs( var ) )
          self.maxabsweight = tf.maximum( maxval, self.maxabsweight )
      self.avgweight = weightsum / tf.to_float( nrweights )
          
      regularization_term = tf.zeros( 1, dtype = tf.float32, name = None )
      nrvalues = tf.zeros( 1, dtype = tf.int32, name = None )
      for weight in self.weights:
        if args.l2_weight_regularization:
          regularization_term = regularization_term +  tf.nn.l2_loss( weight ) 
          nrvalues = tf.add( nrvalues, tf.reduce_prod( tf.shape( weight ) ) )
        if args.max_weight_regularization:
          regularization_term = tf.maximum( regularization_term,  tf.reduce_max( weight ) )
      if args.l2_weight_regularization:
        regularization_term = tf.div( regularization_term, nrvalues )        
      return args.regularization_factor * regularization_term

  def get_stroke_loss( self, args, z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, z_eos, z_eod, x1_data, x2_data, eos_data, eod_data, targetdata_classvars ):

    self.mask = tf.sign( tf.abs( tf.reduce_max( targetdata_classvars, reduction_indices = 1 ) ) )
    self.result0 = tf.squeeze( self.tf_2d_normal( x1_data, x2_data, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr ) ) #batch_nrpoints x nrmixtures   
           
    # implementing eq # 26 of http://arxiv.org/abs/1308.0850
    epsilon = 1e-10
    self.result1 = tf.multiply( self.result0, z_pi )
    self.lossvector = self.result1
    self.result1 = tf.reduce_sum( self.result1, 1, keep_dims = True ) #batch_nrpoints x 1
    self.result1 = tf.squeeze( -tf.log( self.result1 + epsilon ) ) # at the beginning, some errors are exactly zero.
    self.result1_nomask = self.result1
    
    eos_data =  tf.squeeze( eos_data )
    self.z_eos = z_eos
    self.eos_data = eos_data
    self.result2 = tf.multiply( z_eos, eos_data ) + tf.multiply( 1 - z_eos, 1 - eos_data ) #eq 26 rightmost part
    self.result2 = -tf.log( self.result2 + epsilon ) 

    eod_data =  tf.squeeze( eod_data )
    self.result3 = tf.multiply( z_eod, eod_data ) + tf.multiply( 1 - z_eod, 1 - eod_data ) #analogous for eod
    self.result3 = -tf.log( self.result3 + epsilon ) 
            
    self.result = self.result1 + self.result2 + self.result3 

    self.result_before_mask = self.result
    self.result *= self.mask #checked EdJ Oct 15: correctly applies mask to include loss for used points only, depending on current sequence length
    
    self.lossnrpoints = tf.reduce_sum( self.mask )

    stroke_loss = tf.reduce_sum( self.result ) / self.lossnrpoints
    return stroke_loss

  def get_class_loss( self, args, z_classvars, z_classpred, targetdata_classvars ):
    self.mask = tf.sign( tf.abs( tf.reduce_max( targetdata_classvars, reduction_indices = 1 ) ) )

    self.result4 = tf.zeros( 1, dtype = tf.float32, name = None )
    if args.nrClassOutputVars > 0 and args.classweightfactor > 0:
      self.crossentropy = tf.nn.softmax_cross_entropy_with_logits( z_classvars, targetdata_classvars )
      self.result4 = args.classweightfactor *  self.crossentropy 
      self.result4 = tf.multiply( self.mask, self.result4 )
      self.targetdata_classvars = targetdata_classvars
      
    self.result = self.result4

    self.result_before_mask = self.result
    self.result *= self.mask #checked EdJ Sept 2: correctly only measures loss up to last point of actual sequence.
    self.lossvector = self.result

    self.lossnrpoints = tf.reduce_sum( self.mask )
                         
    classloss = tf.reduce_sum( self.result  ) / self.lossnrpoints
    return classloss
  
  def __init__( self, args, trainpredictmode, infer = False, nrinputvars_network = 1, nroutputvars_raw = 1, nrtargetvars = 1, nrauxoutputvars = 0, rangemin = 0, rangelen = 1, maxdigitlength_nrpoints = 1 ):
    self.args = args

    self.result0 = tf.zeros( 1, dtype = tf.float32, name = None )
    self.result1 = tf.zeros( 1, dtype = tf.float32, name = None )
    self.result1_nomask = tf.zeros( 1, dtype = tf.float32, name = None )
    self.result2 = tf.zeros( 1, dtype = tf.float32, name = None )
    self.result3 = tf.zeros( 1, dtype = tf.float32, name = None )
    self.result4 = tf.zeros( 1, dtype = tf.float32, name = None )
    self.crossentropy = tf.zeros( 1, dtype = tf.float32, name = None )
    self.lossvector = tf.zeros( 1, dtype = tf.float32, name = None )
    self.targetdata_classvars = tf.zeros( 1, dtype = tf.float32, name = None )

    self.nrinputvars_network = nrinputvars_network
    self.nroutputvars_raw = nroutputvars_raw
    self.nrauxoutputvars = nrauxoutputvars
    self.maxdigitlength_nrpoints = maxdigitlength_nrpoints
    self.max_seq_length = args.max_seq_length 
    self.seq_length = min( self.max_seq_length, self.maxdigitlength_nrpoints )
    self.regularization_term = tf.zeros( 1, dtype = tf.float32, name = None )
    o_classvars = tf.zeros( 2, dtype = tf.float32, name = None )
    o_classpred = tf.zeros( 2, dtype = tf.float32, name = None )

    if infer:
      self.seq_length = 2 #will be reduced by 1
    
    self.batch_size_ph = tf.placeholder( dtype = tf.int32 )
    self.seq_length_ph = tf.placeholder( dtype = tf.int32 )

    if args.model == 'rnn':
      cell_fn = tf.nn.rnn_cell.BasicRNNCell
    elif args.model == 'gru':
      cell_fn = tf.nn.rnn_cell.GRUCell
    elif args.model == 'basiclstm':
      cell_fn = tf.nn.rnn_cell.BasicLSTMCell
    elif args.model == 'lstm':
      cell_fn = tf.nn.rnn_cell.LSTMCell
    elif args.model == 'ffnn':
      cell_fn = 0
    else:
      raise Exception( "model type not supported: {}".format( args.model ) )

    useInitializers = False
    if hasattr( args, 'useInitializers' ):
      useInitializers = args.useInitializers

    if args.model == 'ffnn': #regular variables, no rnn
      nrinputs = nrinputvars_network
      nrhidden = args.rnn_size
      nroutputs = self.nroutputvars_raw

      if useInitializers:
        self.init_op_weights_ffnn = tf.random_normal( [ nrinputs, nrhidden ], dtype = tf.float32, name = None, seed = random.random( ) )
        init_op_bias_ffnn = tf.zeros( [ nrhidden ], dtype = tf.float32, name = None )
      if args.num_layers > 0:
        if useInitializers:
          weightsh1 = tf.get_variable( "weightsh1", initializer = self.init_op_weights_ffnn ) 
          biasesh1 = tf.get_variable( "biasesh1",  initializer = init_op_bias_ffnn )
        else:
          weightsh1 = tf.get_variable( "weightsh1", [ nrinputs, nrhidden ] ) 
          biasesh1 = tf.get_variable( "biasesh1", [ nrhidden ] )
          
      if args.num_layers > 1:
        if useInitializers:
          weightsh2 = tf.get_variable( "weightsh2", initializer = self.init_op_weights_ffnn ) 
          biasesh2 = tf.get_variable( "biasesh2", initializer = init_op_bias_ffnn )
        else:
          weightsh2 = tf.get_variable( "weightsh2", [ nrhidden, nrhidden ] ) 
          biasesh2 = tf.get_variable( "biasesh2", [ nrhidden ] )                
      layers = tf.zeros( [ 1 ] )
      
    else:
      if args.model == 'lstm':
        layers = cell_fn( args.rnn_size, use_peepholes = True )
      else:
        layers = cell_fn( args.rnn_size )
        
      if args.num_layers > 0:

        rnn_layers= []
        for li in range( args.num_layers ):
            if args.model == 'lstm':
                layer = cell_fn(args.rnn_size, use_peepholes=True)
            else:
                layer = cell_fn(args.rnn_size)

            rnn_layers.append(layer)
        layers = tf.contrib.rnn.MultiRNNCell(cells=rnn_layers, state_is_tuple=True)

      else:
          if args.model == 'lstm':
              layers = cell_fn(args.rnn_size, use_peepholes=True)
          else:
              layers = cell_fn(args.rnn_size)

    if ( infer == False and args.keep_prob < 1 ): # training mode
      layers = tf.nn.rnn_cell.DropoutWrapper( layers, output_keep_prob = args.keep_prob )

    self.layers = layers

    if infer:
      self.input_data = tf.placeholder( dtype = tf.float32, shape = [ None, 1, nrinputvars_network ] )
      self.target_data = tf.placeholder( dtype = tf.float32, shape = [ None, 1, nrtargetvars ] )
    else:
      self.input_data = tf.placeholder( dtype = tf.float32, shape = [ None, self.seq_length - 1, nrinputvars_network ] )
      self.target_data = tf.placeholder( dtype = tf.float32, shape = [ None, self.seq_length - 1, nrtargetvars ] )
    self.batch_size_ph = tf.placeholder( tf.int32, [] )

    if args.model == "ffnn":
      self.initial_state = tf.zeros( [ 1 ] )
    else:
      self.initial_state = state = layers.zero_state( batch_size = self.batch_size_ph, dtype = tf.float32 )

    seqlen = self.seq_length - 1
    self.inputdatasize = tf.shape( self.input_data )
    inputs = tf.split( self.input_data, seqlen, 1)
    self.inputssize1 = tf.shape( inputs )
    inputs = [ tf.squeeze( input_, [ 1 ] ) for input_ in inputs ]
    self.inputssize2 = tf.shape( inputs )

    if useInitializers:
      self.init_op_weights = tf.random_normal( [ args.rnn_size, self.nroutputvars_raw ], dtype = tf.float32, name = None, seed = random.random( ) )
      init_op_bias = tf.zeros( [ self.nroutputvars_raw ], dtype = tf.float32, name = None )
    with tf.variable_scope( trainpredictmode ):
      if useInitializers:
        outputWeight = tf.get_variable( "outputWeight", initializer = self.init_op_weights )
        outputBias = tf.get_variable( "outputBias", initializer = init_op_bias )
      else:
        outputWeight = tf.get_variable( "outputWeight", [ args.rnn_size, self.nroutputvars_raw ] )
        outputBias = tf.get_variable( "outputBias", [ self.nroutputvars_raw ] )
      self.outputWeight = outputWeight
      self.outputBias = outputBias


    if args.model == 'ffnn': #regular variables, no rnn
      print( 'nrinputvars_network', nrinputvars_network )
      inputs_2d = tf.reshape( inputs, [ -1, nrinputvars_network ] ) # make 2d: ( nrseq * seq_length ) x nrinputvars_network    
      
      if args.num_layers > 0:
        hidden1 = tf.nn.relu( tf.matmul( inputs_2d, weightsh1 ) + biasesh1 )
        output = hidden1
      if args.num_layers > 1:
        hidden2 = tf.nn.relu( tf.matmul( output, weightsh2 ) + biasesh2 ) 
        output = hidden2
      last_state = tf.zeros( [ 1 ] )
    elif args.usernn: #See https://www.tensorflow.org/versions/r0.10/tutorials/recurrent/index.html
      output, last_state = tf.contrib.rnn.static_rnn( layers, inputs, initial_state = self.initial_state, scope = trainpredictmode )
    else:
      output, last_state = tf.nn.seq2seq.rnn_decoder( inputs, self.initial_state, layers, loop_function = None, scope = trainpredictmode )

    output = tf.reshape( tf.concat( output, 1 ), [ -1, args.rnn_size ] )
    output = tf.nn.xw_plus_b( output, outputWeight, outputBias )

    self.num_mixture = args.num_mixture
    self.output = output
    self.final_state = last_state

    # reshape target data so that it is compatible with prediction shape
    flat_target_data = tf.reshape( self.target_data, [ -1, nrtargetvars ] ) # make 2d: ( nrseq * seq_length ) x nrinputvars_network    
    targetdata_classvars = flat_target_data[ :, :self.nrauxoutputvars ]
    [ x1_data, x2_data, eos_data, eod_data ] = tf.split( flat_target_data[ :, self.nrauxoutputvars: ], 4, 1 ) #classvars dx dy eos eod

    loss = tf.zeros( 1, dtype = tf.float32, name = None )
    if args.nrClassOutputVars > 0 and args.classweightfactor > 0:
      [ o_classvars, o_classpred ] = self.get_classvars( args, output ) #does same as when strokevars are used, but skips extracting those
      classloss = self.get_class_loss( args, o_classvars, o_classpred, targetdata_classvars )
      loss += classloss
    if args.useStrokeOutputVars and args.useStrokeLoss:
        [ o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, o_eos, o_eod ] = self.get_mixture_coef( args, output )

        self.pi = o_pi
        self.mu1 = o_mu1
        self.mu2 = o_mu2
        self.sigma1 = o_sigma1
        self.sigma2 = o_sigma2
        self.corr = o_corr
        self.eos = o_eos
        self.eod = o_eod
        strokeloss = self.get_stroke_loss( args, o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, o_eos, o_eod, x1_data, x2_data, eos_data, eod_data, targetdata_classvars )
        loss += strokeloss

    self.loss_plain = loss
    self.regularization_term = self.getRegularizationTerm( args )
    loss += self.regularization_term    
    self.loss_total = loss
    
    self.classvars = o_classvars
    self.classpred = o_classpred 

    self.learningratevar = tf.Variable( 0.0, trainable = False )
    self.learningrate_ph = tf.placeholder( dtype = tf.float32 ) #placeholder to feed new values for the learning rate to avoid adding an assignment op for each change
    self.learningrateop = tf.assign( self.learningratevar, self.learningrate_ph )

    tvars = tf.trainable_variables( )

    with tf.variable_scope( "gradient" ):
      self.gradient_org = tf.gradients( loss, tvars )
      self.gradient_clipped, _ = tf.clip_by_global_norm( self.gradient_org, args.grad_clip )    
      optimizer = tf.train.AdamOptimizer( self.learningratevar, epsilon = 1e-05 )
      
      self.train_op = optimizer.apply_gradients( zip( self.gradient_clipped, tvars ) )

  def sample( self, sess, dataloader, args, nrbatches, use_own_output_as_input, outputdir ): #to see how network behaves given perfect prediction by itself on each previous step, feed input so that we can see output on each step
    print( 'sample' )

    fn = outputdir +"output.txt"
    outputfile = open( fn, "w" )
    
    prev_state = sess.run( self.layers.zero_state( 1, tf.float32 ) )

    nrpointsperseq = args.maxdigitlength_nrpoints

    nrpoints = int ( nrbatches * args.nrseq_per_batch * nrpointsperseq )
    nrsequenceinputs = 4 #dx dy eos eod
    strokes = np.zeros( ( nrpoints, nrsequenceinputs ), dtype = np.float32 )
    mixture_params = [ ]

    dataloader.reset_batch_pointer( args )
    state = sess.run( self.initial_state, feed_dict = { self.batch_size_ph: args.nrseq_per_batch, self.seq_length_ph: self.seq_length } )

    strokeindex = 0
    sequencenr = 0
    nrseq = dataloader.curnrexamples
    rmse_strokes = np.zeros( ( nrseq ), dtype = np.float32 )
    rmse_classes = np.zeros( ( nrseq ), dtype = np.float32 )
    correctfracs = np.zeros( ( nrseq ), dtype = np.float32 )

    mode = "test"
    nrbatches = 100
    sample_nrseq = args.nrseq_per_batch

    if use_own_output_as_input:
      sample_nrseq = 500

    for batchnr in range( nrbatches ):
      print( 'batch', batchnr )
      x, y, sequence_index = dataloader.next_batch( args, args.seq_length )

      for batch_seqnr in range( sample_nrseq ):
       print( 'batch', batchnr, 'seq', batch_seqnr, 'of', sample_nrseq, 'filenr', sequence_index[ batch_seqnr ] )
       xseq = x[ batch_seqnr ]
       yseq = y[ batch_seqnr ]
       
       nrpoints = min( len( xseq ), nrpointsperseq )      
       outputmat = np.zeros( ( nrpoints, args.nroutputvars_final ), dtype = np.float32 )

       if use_own_output_as_input:
         maxnrrows = 100
       else:
         maxnrrows = nrpoints
       rownr = 0
       cont = True
       while cont:
        if ( not use_own_output_as_input ) or ( rownr == 0 ):
          inputrow = xseq[ rownr, : ]
          print( 'getting row', rownr, ' of inputdata:', inputrow )
        else:
          inputrow = [ next_x1, next_x2, eos, eod ]

        inputrow_scaledback = np.copy( inputrow )
        inputrow_scaledback[ 0:2 ] *= args.rangelen
        inputrow_scaledback[ 0:2 ] += args.rangemin
        
        print( 'feeding inputrow, scaled back:', inputrow_scaledback )
        feed = {self.input_data: [ [ inputrow ] ], self.initial_state:prev_state}

        [ o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, o_eos, o_eod, o_classvars, o_classpred, next_state, output ] = sess.run( [ self.pi, self.mu1, self.mu2, self.sigma1, self.sigma2, self.corr, self.eos, self.eod, self.classvars, self.classpred, self.final_state, self.output ], feed )
        prev_state = next_state

        if args.nrClassOutputVars > 0:
          classvars = o_classvars[ 0, ]
          classpred = o_classpred[ 0, ]
        
          batch_pointnr = 0

          targetmat = np.copy( yseq )
    
          absrowsum = np.absolute( targetmat ).sum( 1 )
          mask = np.sign( absrowsum )
          nzrows = np.nonzero( mask )
          nzrows = nzrows[ 0 ]
          if len( nzrows )>0:
            last = len( nzrows ) - 1
            nrtargetrows = nzrows[ last ] + 1
          else:
              nrtargetrows = 0
          print( 'found nrtargetrows:', nrtargetrows )
        
          outputmat = np.zeros( ( 1, args.nroutputvars_final ), dtype = np.float32 )
          outputmat_sampled = np.zeros( ( 1, args.nroutputvars_final ), dtype = np.float32 )
      
          if args.useStrokeOutputVars:
            if args.nrClassOutputVars > 0 and args.classweightfactor > 0:                
              outputmat[ 0, :args.nrClassOutputVars ] = o_classpred[ batch_pointnr, ]
              outputmat_sampled[ 0, :args.nrClassOutputVars ] = o_classpred[ batch_pointnr, ]
            if args.useStrokeLoss:
              idx = get_pi_idx( dataloader.getRandValue( ), o_pi[ batch_pointnr ] )
              next_x1, next_x2 = self.sample_gaussian_2d( o_mu1[ batch_pointnr, idx ], o_mu2[ batch_pointnr, idx ], o_sigma1[ batch_pointnr, idx ], o_sigma2[ batch_pointnr, idx ], o_corr[ batch_pointnr, idx ] )
              eos = 1 if dataloader.getRandValue( ) < o_eos[ batch_pointnr ] else 0         
              eod = 1 if dataloader.getRandValue( ) < o_eod[ batch_pointnr ] else 0
              outputmat[ 0, args.nrClassOutputVars:args.nrClassOutputVars + 4 ] = [ o_mu1[ batch_pointnr, idx ], o_mu2[ batch_pointnr, idx ], o_sigma1[ batch_pointnr, idx ], o_sigma2[ batch_pointnr, idx ] ]
              outputmat_sampled[ 0, args.nrClassOutputVars:args.nrClassOutputVars+4 ] = [ next_x1, next_x2, eos, eod ]
          else:
            outputmat_sampled[ 0, ] = o_classpred[ batch_pointnr, ]

          print( 'output unscaled:', [ o_mu1[ batch_pointnr, idx ], o_mu2[ batch_pointnr, idx ], o_eos[ batch_pointnr ], o_eod[ batch_pointnr ] ] )
          outputrow = np.asarray( [ next_x1, next_x2, eos, eod ] )
          print( 'sampled output unscaled:', outputrow )
          outputrow[ 0:2 ] *= args.rangelen
          outputrow[ 0:2 ] += args.rangemin
          print( 'sampled output scaled', outputrow )
          outputfile.write( str( outputrow[ 0 ] ) + " " + str( outputrow[ 1 ] ) + " "  + str( outputrow[ 2 ] ) + " "  + str( outputrow[ 3 ] ) + "\n" )

          if not use_own_output_as_input:
           stroketarget = np.copy( targetmat[ rownr, args.nrClassOutputVars:args.nrClassOutputVars + 2 ] )    
           classtarget = np.copy( targetmat[ rownr, :args.nrClassOutputVars ] )
           print( 'classtarget:', classtarget )
        
           if args.useStrokeOutputVars: 
                        
            if args.useStrokeLoss:  
              outputmat_sampled[ :, args.nrClassOutputVars:args.nrClassOutputVars + 2 ] *= args.rangelen
              outputmat_sampled[ :, args.nrClassOutputVars:args.nrClassOutputVars + 2 ] += args.rangemin
              outputmat[ :, args.nrClassOutputVars:args.nrClassOutputVars + 2 ] *= args.rangelen
              outputmat[ :, args.nrClassOutputVars:args.nrClassOutputVars + 2 ] += args.rangemin

              print( 'sampled outputmat_sample scaled back:' )
              print( outputmat_sampled )

              stroketarget *= args.rangelen
              stroketarget += args.rangemin
    
              err_stroke = outputmat_sampled[ :, args.nrClassOutputVars:args.nrClassOutputVars + 2 ]-stroketarget
    
              print( 'prediction', mode )
              print( outputmat_sampled[ :, args.nrClassOutputVars:args.nrClassOutputVars + 2 ] )
              print( 'stroketarget', mode )
              print( stroketarget )
              print( 'error', mode )
              print( err_stroke )
                            
              sse_stroke = ( err_stroke ** 2 ).sum( )
                      
          if args.nrClassOutputVars > 0 and not use_own_output_as_input:
              classindex_true = np.argmax( classtarget )
              classindex_pred = np.argmax( outputmat_sampled[ 0, :args.nrClassOutputVars ] )
              print( 'batch', batchnr, 'seq', batch_seqnr, 'row', rownr, "classindex_true", classindex_true, 'pred', classindex_pred )
              class_logits = outputmat_sampled[ 0, :args.nrClassOutputVars ]

              correct = np.equal( classindex_pred, classindex_true )
    
              print( 'output', outputmat_sampled[ :args.nrClassOutputVars ] )
              last = args.nroutputvars_raw - args.nrClassOutputVars
              logits_str = [ str( a ) for a in class_logits ]
              print( "batch", batchnr, "class", classindex_true, 'pred', classindex_pred, "class_logits", " " . join( logits_str ) )
              print( 'correct:', correct )
        
        rownr += 1
        if eod or ( rownr >= maxnrrows ):
          cont = False
          prev_state = sess.run( self.layers.zero_state( 1, tf.float32 ) )

      print( 'end of batch', batchnr )
    print( 'done' ) #after batch for loop
    outputfile.close( )



