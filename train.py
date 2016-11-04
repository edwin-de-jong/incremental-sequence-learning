import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops

import getpass
import argparse
import time
import os
import numpy.ma as MA
import sys
import pickle
import random
import math
from time import gmtime, strftime

from utils import DataLoader
import utils

from model import Model
from model import get_pi_idx

from stats import Stats

import resource

def memusage( point = "") :
    usage = resource.getrusage( resource.RUSAGE_SELF) 
    return '''%s: usertime = %s systime = %s mem = %s mb
           '''%( point, usage[ 0 ], usage[ 1 ], 
                ( usage[ 2 ]*resource.getpagesize( ) ) /1000000.0 ) 

def main( ) :
  
  print_input = 0
  user = getpass.getuser( ) 
  print ( "user: ", user) 
  logdir = "/home/"+user+"/code/digits/rnn/log/"

  parser = argparse.ArgumentParser( ) 
  parser.add_argument( '--rnn_size', type = int, default = 256, 
                     help = 'size of RNN hidden state') 
  parser.add_argument( '--num_layers', type = int, default = 2, 
                     help = 'number of layers in the RNN') 
  parser.add_argument( '--model', type = str, default = 'basiclstm', 
                      help = 'rnn, gru, or lstm, or ffnn') 
  parser.add_argument( '--nrseq_per_batch', type = int, default = 50, 
                     help = 'minibatch size') 
  parser.add_argument( '--nrseq_per_batch_test', type = int, default = 50, 
                     help = 'minibatch size') 
  parser.add_argument( '--nrpoints_per_batch', type = int, default = 0, 
                     help = 'Number of points ( sequence steps)  per batch') 
  parser.add_argument( '--num_epochs', type = int, default = 30, 
                     help = 'number of epochs') 
  parser.add_argument( '--report_every', type = int, default = 50, 
                     help = 'report frequency') 
  parser.add_argument( '--save_every_nrbatches', type = int, default = 200, 
                     help = 'save frequency') 
  parser.add_argument( '--save_maxnrmodels_keep', type = int, default = 5, 
                     help = 'Max nr of models to keep') 
  parser.add_argument( '--eval_every', type = int, default = 50, 
                     help = 'evaluation frequency') 
  parser.add_argument( '--test_every_nrbatches', type = int, default = 0, 
                     help = 'testing frequency') 
  parser.add_argument( '--grad_clip', type = float, default = 10., 
                     help = 'clip gradients at this value') 
  parser.add_argument( '--learning_rate', type = float, default = 0.005, 
                     help = 'learning rate') 
  parser.add_argument( '--decay_rate', type = float, default = 0.95, 
                     help = 'decay rate for rmsprop') 
  parser.add_argument( '--num_mixture', type = int, default = 20, 
                     help = 'number of gaussian mixtures') 
  parser.add_argument( '--keep_prob', type = float, default = 0.8, 
                     help = 'dropout keep probability') 
  parser.add_argument( '--predict', type = int, default = 0, 
                     help = 'predict instead of training') 
  parser.add_argument( '--predictideal', type = int, default = 0, 
                     help = 'predict given ideal input') 
  parser.add_argument( '--evaluate', type = int, default = 0, 
                     help = 'Run evaluation process that monitors the checkpoint files of a concurrently running training process.') 
  parser.add_argument( '--nrinputfiles_train', type = int, default = 0, 
                     help = 'number of training input data files to use') 
  parser.add_argument( '--nrinputfiles_test', type = int, default = 0, 
                     help = 'number of test input data files to use') 

  parser.add_argument( '--explabel', type = str, default = 0, 
                     help = 'experiment label') 
  parser.add_argument( '--max_seq_length', type = int, default = 0, 
                     help = 'max amount of points per sequence that will be used') 
  parser.add_argument( '--file_label', type = str, default = "", 
                     help = 'input file label') 
  parser.add_argument( '--train_on_own_output_method', type = int, default = 0, 
                     help = 'Various methods for training on own output, governed by current network performance') 
  parser.add_argument( '--model_checkpointfile', type = str, default = "", 
                     help = 'checkpoint file to load') 
  parser.add_argument( '--sample_from_output', type = int, default = 0, 
                     help = 'If set, when using train_on_own_output_method, the output will be sampled from first before passing it on as the next input; if not set, the output is used directly.') 
  parser.add_argument( '--regularization_factor', type = float, default = .01) 
  parser.add_argument( '--l2_weight_regularization', type = int, default = 1, help = 'Use the average of all weights as a regularization component') 
  parser.add_argument( '--max_weight_regularization', type = int, default = 0, help = 'Use the maximum of all weights as a basis for regularization') 
  parser.add_argument( '--discard_inputs', type = int, default = 0, help = 'Discard the input data; network must produce the output balistically.') 
  parser.add_argument( '--discard_classvar_inputs', type = int, default = 0, help = 'Discard the 10 boolean class indicators indicating the current class as input.') 
  parser.add_argument( '--nrClassOutputVars', type = int, default = 0, help = 'Use up to 10 binary class indicator outputs that the model has to predict at each step') 
  parser.add_argument( '--useStrokeOutputVars', type = int, default = 1, help = 'Generate strokes.') 
  parser.add_argument( '--useStrokeLoss', type = int, default = 1, help = 'Use loss component based on stroke output.') 
  parser.add_argument( '--useClassInputVars', type = int, default = 1, help = 'Use 10 binary input variable representing the digit class ( one-hot representation) .') 
  parser.add_argument( '--incremental_min_nrpoints', type = int, default = 5000, help = 'Min number of points to evaluate before considering next increment') 
  parser.add_argument( '--useInitializers', type = int, default = 0, help = 'Use initializers for network parameters to ensure reproducibility') 
  parser.add_argument( '--usePreviousEndState', type = int, default = 0, help = 'Use end state after previous batch as initial state for next batch') 
  parser.add_argument( '--print_length_correct', type = int, default = 0, help = 'Use initializers for network parameters to ensure reproducibility') 
  
  parser.add_argument( '--startingpoint', type = str, default = '', help = 'Start from saved state.') 
  parser.add_argument( '--randomizeSequenceOrder', type = int, default = 1, help = 'Randomize order of sequences to prevent learning order.') 
  parser.add_argument( '--classweightfactor', type = float, default = 10, help = 'weight of classvar loss') 
  parser.add_argument( '--curnrtrainexamples', type = float, default = 10) 
  parser.add_argument( '--current_seq_length', type = int, default = 0, 
                     help = 'Used in combination with incremental_seq_length') 
  parser.add_argument( '--curnrdigits', type = int, default = 10) 
  parser.add_argument( '--correctfrac_threshold_inc_nrtrainex', type = float, default = .8) 
  parser.add_argument( '--threshold_rmse_stroke', type = float, default = 2) 
  parser.add_argument( '--usernn', type = int, default = 0) 
  parser.add_argument( '--fileselection', type = str, default = '', nargs = '+')  #representative 10 digits: 1, 3, 25, 7, 89, 0, 62, 96, 85, 43
  parser.add_argument( '--incremental_nr_trainexamples', type = int, default = 0) 
  parser.add_argument( '--incremental_seq_length', type = int, default = 0) 
  parser.add_argument( '--incremental_nr_digits', type = int, default = 0) 
  parser.add_argument( '--runnr', type = int, default = 1) 
  parser.add_argument( '--maxnrpoints', type = int, default = 0) 
  parser.add_argument( '--stat_windowsize_nrsequences', type = int, default = 1000) 
  parser.add_argument( '--firsttrainstep', type = int, default = 0, help = 'Loss is calculated from this sequence step onwards; preceding points are ignored ( fed, but not contributing to loss) ') 
  parser.add_argument( '--stopcrit_threshold_stroke_rmse_train', type = float, default = 0) 
  parser.add_argument( '--testovertrain', type = int, default = 0, help = 'Control experiment to check that overtraining can occur. Uses digits 0-4 for training, 5-9 for testing.') 
  parser.add_argument( '--reportstate', type = int, default = 0, help = 'report complete internal state ( weights, state)  before/after each train/test batch') 
  parser.add_argument( '--reportmixture', type = int, default = 0, help = 'report mixture') 

  #arguments set internally:
  parser.add_argument( '--maxdigitlength_nrpoints', type = int, default = 0, help = 'max sequence length ( nr points)  that was encountered in the training data; calculated parameter.' ) 
  parser.add_argument( '--rangemin', type = float, default = -22.6)  #determined based on full MNIST stroke sequence data set
  parser.add_argument( '--rangelen', type = float, default = 55.2)  #determined based on full MNIST stroke sequence data set
  parser.add_argument( '--seq_length', type = int, default = 0) 
  parser.add_argument( '--nroutputvars', type = int, default = 0) 
  parser.add_argument( '--nrtargetvars', type = int, default = 0) 
  parser.add_argument( '--nrauxinputvars', type = int, default = 0) 
  parser.add_argument( '--debuginfo', type = int, default = 0) 
  
  #variable sizes:
  #o_pi: nrrowsperbatch x nrmixtures, i.e. pointnr x mixturenr
  #targetdata: nrseq x seqlen x nrinputvars_network
  #input: dx dy eos eod
  #output: eos eod nr_mixtures*distribution-params classvars

  args = parser.parse_args( ) 
  
  file_label = args.file_label

  explabel = args.explabel

  outputdir = "./results/"+explabel+"r"+str( args.runnr) +"/"

  if args.incremental_nr_trainexamples:
      args.curnrtrainexamples = min( args.curnrtrainexamples, args.nrinputfiles_train ) 
      args.incremental_min_nrpoints = 50 * args.curnrtrainexamples
  else:
      args.curnrtrainexamples = args.nrinputfiles_train

  datadir = "/home/"+user+"/code/digits/sequences/"
  print( "using data dir: ", datadir ) 

  seqlenarg = 0
  trainarg = 1
  dataloader_train = DataLoader( datadir, args, args.nrinputfiles_train, args.curnrtrainexamples, seqlenarg, trainarg, file_label, print_input, args.rangemin, args.rangelen) 
  args.nrauxinputvars = 10 * args.useClassInputVars

  trainarg = 0
  dataloader_test = DataLoader( datadir, args, args.nrinputfiles_test, args.nrinputfiles_test, dataloader_train.seq_length, trainarg, file_label, print_input, args.rangemin, args.rangelen) 
  dataloader_test.createRandValues( )         

  args.nrtargetvars = 4*args.useStrokeOutputVars + args.nrClassOutputVars
  if ( not args.incremental_seq_length) :
      args.current_seq_length = dataloader_train.seq_length
  
  if ( args.evaluate or args.predict or args.predictideal) :
    configfile = os.path.join( 'save/'+args.explabel+'r'+str( args.runnr) , 'config.pkl') 
    if len( args.startingpoint )  > 0:
        pos = args.startingpoint.find( 'model' ) 
        savedfolder = args.startingpoint[ 0 : pos - 1 ] 
        print( 'savedfolder', savedfolder) 
        slashpos = savedfolder.find( '/')  #find first slash
        savedfolderlist = list( savedfolder) 
        savedfolderlist[ slashpos ] = 'x'
        savedfolder = "".join( savedfolderlist) 
        slashpos = savedfolder.find( '/')  #find second slash       
        savedfolder = args.startingpoint[ 0 : pos - 1 ]
        savedfolderlist = list( savedfolder) 
        savedfolderlist = savedfolderlist[ :slashpos+1 ]
        savedfolder = "".join( savedfolderlist)  #get the path                                                
        configfile = os.path.join( savedfolder, 'config.pkl') 
        print( 'configfile', configfile) 
    fileexists = os.path.exists( configfile ) 
    while not fileexists:
        print ( "Waiting for config file", configfile) 
        time.sleep( 5) 
        fileexists = os.path.exists( configfile ) 

    f = open( configfile, "rb") 
    saved_args = pickle.load( f) 
    saved_args.nrseq_per_batch = args.nrseq_per_batch
    f.close( ) 
    
    trainpredictmode = "Predict"
    nrsequenceinputs = 4 #dx dy eos eod
    nrinputvars_network = nrsequenceinputs + args.nrauxinputvars;
    
    model_predict = Model( saved_args, trainpredictmode, True, nrinputvars_network, saved_args.nroutputvars_raw, args.nrtargetvars, args.nrClassOutputVars, maxdigitlength_nrpoints = saved_args.maxdigitlength_nrpoints) 

  if args.predict:
    nrbatches = args.nrinputfiles_test / args.nrseq_per_batch
    use_own_output_as_input = 1
    with tf.Session( )  as sess:
        performPrediction( sess, saved_args, args, model_predict, dataloader_test, nrbatches, use_own_output_as_input, outputdir, parser ) 
  elif ( args.predictideal) :
    nrbatches = args.nrinputfiles_test / args.nrseq_per_batch
    use_own_output_as_input = 0
    with tf.Session( )  as sess:
        performPrediction( sess, saved_args, args, model_predict, dataloader_test, nrbatches, use_own_output_as_input, outputdir, parser ) 
  else:      
      train( dataloader_train, dataloader_test, args, logdir, outputdir ) 

def savemodel( saver, sess, dataloader, args, batchnr ) :
    checkpoint_path = os.path.join( 'save/'+args.explabel+'r'+str( args.runnr) , 'model.ckpt') 
    print( ( "saving model to {}".format( checkpoint_path) ) ) 
    saver.save( sess, checkpoint_path, global_step = batchnr) 
    checkpoint_fullpath = checkpoint_path + "-" + str( batchnr ) 
    print( ( 'saved checkpoint: '+checkpoint_fullpath) ) 
    
def restoreModel( sess, args, model_predict, dataloader, checkpoint_fullpath = "") :

    saver = tf.train.Saver( tf.trainable_variables( ) , max_to_keep = args.save_maxnrmodels_keep) 

    print( 'restoreModel: checkpoint_fullpath', checkpoint_fullpath) 
    if len( checkpoint_fullpath) >0: #model path provided
        saver.restore( sess, checkpoint_fullpath) 
    else: #load most recent state of own run
        modelfile = 'save/'+saved_args.explabel
        if len( args.model_checkpointfile)  > 1 : 
            modelfile = args.model_checkpointfile 
        ckpt = tf.train.get_checkpoint_state( modelfile) 
        print( 'restored checkpoint' ) 
        print ( "loading model: ", ckpt.model_checkpoint_path) 
        saver.restore( sess, ckpt.model_checkpoint_path) 
      
def performPrediction( sess, saved_args, args, model_predict, dataloader, nrbatches, use_own_output_as_input, outputdir, parser) :

    print( 'performprediction') 
    
    saver = tf.train.Saver( tf.trainable_variables( ) , max_to_keep = args.save_maxnrmodels_keep) 

    restoreModel( sess, args, model_predict, dataloader, args.startingpoint) 
    
    print( 'restored model') 

    [ strokes, params ] = model_predict.sample( sess, dataloader, saved_args, nrbatches, use_own_output_as_input, outputdir) 
    print( 'Completed performPrediction' ) 

def writeOutputTarget( args, outputdir, batchnr, sequence_index, batch_seqnr, outputmat, outputmat_sampled, targetmat, stroketarget, lossvec, model, loss, mode, inputdata) :
  fn = outputdir + "output-"+mode+"-batch-" + str( batchnr)  + "-seqnr-"+str( batch_seqnr)  +"-filenr-" + str( sequence_index[ batch_seqnr ] )  + ".txt" 
  np.savetxt( fn, outputmat, fmt = '%.3f') 

  fn = outputdir + "output-sampled-"+mode+"-batch-" + str( batchnr)  + "-seqnr-"+str( batch_seqnr)  +"-filenr-" + str( sequence_index[ batch_seqnr ] )  + ".txt" 
  np.savetxt( fn, outputmat_sampled, fmt = '%.3f') 

  fn = outputdir + "classtarget-" +mode+ "-batch-" + str( batchnr)  + "-seqnr-" +str( batch_seqnr)  +"-filenr-" + str( sequence_index[ batch_seqnr ] )  +  ".txt"
  np.savetxt( fn, targetmat[ :, 0:10 ], fmt = '%.3f') 

  fn = outputdir + "stroketarget-"+ mode + "-batch-" + str( batchnr)  + "-seqnr-" +str( batch_seqnr)  +"-filenr-" + str( sequence_index[ batch_seqnr ] )  +  ".txt"
  np.savetxt( fn, stroketarget, fmt = '%.3f') 

  fn = outputdir + "input-"+ mode + "-batch-" + str( batchnr)  + "-seqnr-" +str( batch_seqnr)  +"-filenr-" + str( sequence_index[ batch_seqnr ] )  +  ".txt"
  np.savetxt( fn, inputdata, fmt = '%.3f') 

  if batch_seqnr == 0: 
    if ( args.useStrokeOutputVars and args.useStrokeLoss) :
      fn = outputdir + "lossvec-" +mode+ str( batchnr)  + ".txt" 
      np.savetxt( fn, lossvec, fmt = '%.3f')     
      fn = outputdir + mode+"loss-" +mode+ str( batchnr)  + ".txt"
      file = open( fn, "w") 
      file.write( str( loss)  + "\n" ) 
      file.close( ) 
    
def writeMixture( args, outputdir, batchnr, sequence_index, batch_seqnr, mode, mixture, seq_pointnr) :
    fn = outputdir + "mixture-"+mode+"-batch-" + str( batchnr)  + "-seqnr-"+str( batch_seqnr)  +"-filenr-" + str( sequence_index[ batch_seqnr ] )  + ".txt"
    if seq_pointnr == 0:
        f = open( fn, 'wb') 
    else:
        f = open( fn, 'ab') 
    np.savetxt( f, mixture[ None ], fmt = '%.3f', delimiter = ", ") 
    f.close( ) 
          
def softmax( x) :
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp( x - np.max( x) ) 
    return e_x / e_x.sum( ) 

def evaluate( sess, args, stats, stats_alldata, stats_inc, sequence_index, trainpredictmode, model, dataloader, outputdir, outputs, state, lossvec, train_loss, regularization_term, loss_plain, loss_total, weights, nrinputvars_network, targetdata, maxabsweight, avgweight, learningrate_value, train_on_output, epochnr, totbatchnr, totnrpoints_trained, writefiles, runtime, mode, printstate, batchsize_nrseq, x) :    
  nanfound = False
  avgmu1 = 0
  avgmu2 = 0
  maxabscorr = 0
  sse_stroke = 0
  nrrowsused = 0
  report_nrsequences = 10
  
  with tf.variable_scope( trainpredictmode) :

    if args.useStrokeOutputVars and args.useStrokeLoss:  
        [ o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, o_eos, o_eod, o_classvars, o_classpred ] = outputs        
    else:
        [ o_classvars, o_classpred ] = outputs

    batch_pointnr = 0

    for batch_seqnr in range( 0, batchsize_nrseq ) : #for each sequence in the batch
      targetmat = np.copy( targetdata[ batch_seqnr ]) 
      absrowsum = np.absolute( targetmat) .sum( 1 ) 
      mask = np.sign( absrowsum) #checked EdJ Sept 2
      nzrows = np.nonzero( mask)  #not binary; seems: this returns indices of nonzero rows
      nzrows = nzrows[ 0 ]
      if len( nzrows) >0:
          last = len( nzrows)  - 1
          nrtargetrows = nzrows[ last ] + 1
      else:
          nrtargetrows = 0

      if mode == "train":
          evalseqlength = min( args.current_seq_length + 1, nrtargetrows + 1) 
      else:
          evalseqlength = min( model.seq_length, nrtargetrows + 1) 

      outputmat = np.zeros( ( evalseqlength - 1, args.nroutputvars_final) , dtype = np.float32) 
      outputmat_sampled = np.zeros( ( evalseqlength - 1, args.nroutputvars_final) , dtype = np.float32) 

      mixture = 0
      for p in range( evalseqlength - 1) :#process used points first

        if args.useStrokeOutputVars:
            if args.nrClassOutputVars > 0 and args.classweightfactor > 0:                
                outputmat[ p, :args.nrClassOutputVars ] = o_classpred[ batch_pointnr, ]
                outputmat_sampled[ p, :args.nrClassOutputVars ] = o_classpred[ batch_pointnr, ]
            if args.useStrokeLoss:
              idx = get_pi_idx( dataloader.getRandValue( ) , o_pi[ batch_pointnr ]) 
              next_x1, next_x2 = model.sample_gaussian_2d( o_mu1[ batch_pointnr, idx ], o_mu2[ batch_pointnr, idx ], o_sigma1[ batch_pointnr, idx ], o_sigma2[ batch_pointnr, idx ], o_corr[ batch_pointnr, idx ]) 
              eos = 1 if dataloader.getRandValue( )  < o_eos[ batch_pointnr ] else 0         
              eod = 1 if dataloader.getRandValue( )  < o_eod[ batch_pointnr ] else 0
              outputmat[ p, args.nrClassOutputVars:args.nrClassOutputVars+4 ] = [ o_mu1[ batch_pointnr, idx ], o_mu2[ batch_pointnr, idx ], o_sigma1[ batch_pointnr, idx ], o_sigma2[ batch_pointnr, idx ] ]
              outputmat_sampled[ p, args.nrClassOutputVars:args.nrClassOutputVars+4 ] = [ next_x1, next_x2, eos, eod ]
              if writefiles and args.reportmixture and ( sequence_index[ batch_seqnr ] < report_nrsequences) :                     
                  nrparams = args.num_mixture * 6
                  mixture = np.zeros( ( nrparams ) , dtype = np.float32 ) 
                  for m in range( args.num_mixture ) :
                      mixture[ m*6:( m+1) *6 ] = [ o_pi[ batch_pointnr, m ], o_mu1[ batch_pointnr, m ], o_mu2[ batch_pointnr, m ], o_sigma1[ batch_pointnr, m ], o_sigma2[ batch_pointnr, m ], o_corr[ batch_pointnr, m ] ]
                      writeMixture( args, outputdir, totbatchnr, sequence_index, batch_seqnr, mode, mixture, p) 

        else:
            outputmat_sampled[ p, ] = o_classpred[ batch_pointnr, ]

        batch_pointnr += 1
      batch_pointnr += model.seq_length - evalseqlength #after cur seq len, skip to end of seq ( = seq_length - 1) 

      stroketarget = np.copy( targetmat[ :evalseqlength - 1, args.nrClassOutputVars:args.nrClassOutputVars + 4 ])     
      nrrowsused = nrtargetrows      
      stats_inc_rmse = 0

      if args.useStrokeOutputVars and ( nrrowsused > 0) : 
                
        if args.useStrokeLoss:  
          outputmat_sampled[ :, args.nrClassOutputVars:args.nrClassOutputVars + 2 ] *= args.rangelen
          outputmat_sampled[ :, args.nrClassOutputVars:args.nrClassOutputVars + 2 ] += args.rangemin
          outputmat[ :, args.nrClassOutputVars:args.nrClassOutputVars + 2 ] *= args.rangelen
          outputmat[ :, args.nrClassOutputVars:args.nrClassOutputVars + 2 ] += args.rangemin

          stroketarget[ :, 0:2 ] *= args.rangelen
          stroketarget[ :, 0:2 ] += args.rangemin

          err_stroke = outputmat_sampled[ :, args.nrClassOutputVars:args.nrClassOutputVars + 2 ]-stroketarget[ :, 0:2 ] #was: 1:2

          sse_stroke = ( err_stroke ** 2) .sum( ) 

          stats.stats_stroke.log_sse_sequential( sse_stroke, 2 * nrrowsused ) 
          stats_alldata.stats_stroke.log_sse( sequence_index[ batch_seqnr ], sse_stroke, 2 * nrrowsused ) 

          if mode == "train":
              stats_inc.stats_stroke.log_sse_sequential( sse_stroke, 2 * nrrowsused )  #sequential counter; window of last n values
              stats_inc_rmse = stats_inc.stats_stroke.rmse( ) 
              
      if args.nrClassOutputVars > 0 and ( nrrowsused > 0) :
          classindex_true = np.argmax( targetmat[ :evalseqlength - 1, :args.nrClassOutputVars ], 1) 
          classindex_pred = np.argmax( outputmat_sampled[ :, :args.nrClassOutputVars ], 1) 

          correct = np.equal( classindex_pred, classindex_true) 
          last_correct = correct[ nrrowsused - 1 ] #model.seq_length

          if args.print_length_correct:
              seqindex = sequence_index[ batch_seqnr ]
              print( 'len-correct', mode, 'seq', seqindex, 'len', dataloader.seqlengthlist[ seqindex ], 'correct', 1*last_correct) 
              
          stats.stats_correct.log_value_sequential( last_correct, 1 )                             
          stats.stats_correctfrac.log_value_sequential( correct.sum( ) , nrrowsused )                             
          stats_alldata.stats_correct.log_value( sequence_index[ batch_seqnr ], last_correct, 1 )                             
          stats_alldata.stats_correctfrac.log_value( sequence_index[ batch_seqnr ], correct.sum( ) , nrrowsused )                             

      else:
          avgcorrectfrac = 0
          correctpreds = 0

      if writefiles and ( sequence_index[ batch_seqnr ] < report_nrsequences) :
          inputdata = np.copy( x[ batch_seqnr ] ) 
          inputdata[ :, 10 * args.useClassInputVars : 10 * args.useClassInputVars + 2 ] *= args.rangelen
          inputdata[ :, 10 * args.useClassInputVars : 10 * args.useClassInputVars + 2 ] += args.rangemin
          writeOutputTarget( args, outputdir, totbatchnr, sequence_index, batch_seqnr, outputmat, outputmat_sampled, targetmat, stroketarget, lossvec, model, train_loss, mode, inputdata) 

    weights_o = sess.run( model.outputWeight) ;
    bias = sess.run( model.outputBias) ;
    avgbias = bias.mean( ) 
    maxabsbias = np.absolute( bias) .max( ) 
    avgstate = np.asarray( state) .mean( ) 
    maxabsstate = np.absolute( state) .max( ) 
    if args.useStrokeOutputVars and args.useStrokeLoss:  
        avgmu1 = outputmat[ :, 0 ].mean( ) 
        avgmu2 = outputmat[ :, 1 ].mean( ) 
        maxabscorr = np.absolute( o_corr) .max( ) 
        
    avgw = avgweight
    if ( len( avgweight) >1) :
        avgw = avgweight.mean( ) 
    maxabsw = maxabsweight
    if ( len( maxabsweight) >1) :
        maxabsw = maxabsweight.mean( ) 
    print ( 'eval', mode, ': epoch', epochnr, 'totbatches', totbatchnr, 'totnrpoints_trained', totnrpoints_trained, 'nrtrainex', args.curnrtrainexamples, 'curseqlen', args.current_seq_length, 'curnrdigits', args.curnrdigits, 'rmse_stroke', stats.stats_stroke.rmse( ) , 'rmse_stroke_alldata', stats_alldata.stats_stroke.rmse( ) , 'rmse_stroke_inc', stats_inc_rmse, "correct", stats.stats_correct.average( ) , "correct_alldata", stats_alldata.stats_correct.average( ) , 'regularization', regularization_term[ 0 ], 'loss_total', loss_total, 'avgbias', avgbias, 'maxabsbias', maxabsbias, 'avgstate', avgstate, 'maxabsstate', maxabsstate, 'learningrate', learningrate_value, 'maxabscorr', maxabscorr, 'maxabsweight', maxabsw[ 0 ], 'avgweight', avgw[ 0 ], 'runtime', runtime ) 

    #stats
    if epochnr % 100 == 0:
        graph = tf.get_default_graph( ) 
        ops = graph.get_operations( ) 
        print( ( 'mem nr ops: ', len( ops) ) ) 
        print( 'mem usage:') 
        print( ( memusage( "eval") ) ) 
        print( 'rand e', epochnr, dataloader.getRandValue( ) ) 

    return nanfound

    
def print_model( model ) :
    print ( "model structure: " ) 
    print ( "gradient vars: " ) 
     
    for var in tf.get_collection( tf.GraphKeys.VARIABLES, scope = 'gradient') : #    tf.variable_scope( "gradient") :
        print ( "var: ", var.name ) 
    print ( "all vars: " ) 
    params = tf.all_variables( ) 
    for var in params:
        print ( "var: ", var.name ) 
        
def recordState( model, sess ) :
    params = tf.all_variables( ) 
    state = [ ]
    varnames = [ ]
    for var in params:
        varnames.append( var.name ) 
        value = sess.run( var) ;
        state.append( value ) 
    return state, varnames

def printState ( state, varnames, fn = '' ) :
  i = 0
  statefile = open( fn, "w" ) 
  for var in state:
      print( 'var: ', varnames[ i ], file = statefile ) 
      print( var.sum( ) , file = statefile ) 
      i += 1
  statefile.close( ) 
        
def constructInputFromOutput( args, model, x, o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, o_eos, o_eod) :
    xnrseq = np.shape( x) [ 0 ]
    xnrpointsperseq = np.shape( x) [ 1 ]
    getbatch = False
    
    point = 0
    for s in range( xnrseq) : #for each sequence in previous batch

        outputmat = np.zeros( ( model.seq_length, 4) , dtype = np.float32) 

        batch_pointnr = 0
        for p in range( xnrpointsperseq) :
            idx = get_pi_idx( dataloader.getRandValue( ) , o_pi[ batch_pointnr ]) 

            mu1out = o_mu1[ batch_pointnr, idx ] #these are regular numpy floats, not tensors
            mu2out = o_mu2[ batch_pointnr, idx ]
            sigma1out = o_sigma1[ batch_pointnr, idx ]
            sigma2out = o_sigma2[ batch_pointnr, idx ]
            corrout = o_corr[ batch_pointnr, idx ]
            eosout = o_eos[ batch_pointnr ]
            eodout = o_eod[ batch_pointnr ]
            next_x1, next_x2 = model.sample_gaussian_2d( mu1out, mu2out, sigma1out, sigma2out, corrout) 
            eos = 1 if dataloader.getRandValue( )  < eosout else 0         
            eod = 1 if dataloader.getRandValue( )  < eodout else 0
                          
            if args.sample_from_output:
                outputmat[ p, ] = [ next_x1, next_x2, eos, eod ]
            else:
                outputmat[ p, ] = [ mu1out, mu2out, eosout, eodout ]

            batch_pointnr += 1

        fromval = s*xnrpointsperseq
        toval = ( s+1) *xnrpointsperseq-1 #last output: not used
        xvalues = np.array( x[ s ]) 
        xvalues[ 1:xnrpointsperseq, args.nrClassOutputVars:args.nrClassOutputVars+4 ] = outputmat[ 0:xnrpointsperseq-1, ]
        x[ s ] = xvalues
    return x
          
def printInputsTargets( args, x, y ) :
    print ( "x") 
    xvalues = np.array( x) 
    xvalues[ :, args.nrClassOutputVars:args.nrClassOutputVars+2 ] *= args.rangelen
    xvalues[ :, args.nrClassOutputVars:args.nrClassOutputVars+2 ] += args.rangemin
    print( xvalues) 
    print( "y") 
    yvalues = np.array( y) 
    yvalues[ :, args.nrClassOutputVars:args.nrClassOutputVars+2 ] *= args.rangelen
    yvalues[ :, args.nrClassOutputVars:args.nrClassOutputVars+2 ] += args.rangemin
    print( yvalues) 

def printWeightsGradients( sess ) :

    allvars = tf.all_variables( ) 
    for var in allvars:
        isBias = var.name.find( "Bias")  >= 0
        if not isBias:
            print( "var: ", var.name) 
            value = sess.run( var) 
            print( value )                       
              
def train( dataloader_train, dataloader_test, args, logdir, outputdir ) :
    
    stats_train = Stats( args, args.stat_windowsize_nrsequences, 'stats_train' )   #stats over recent training data
    stats_test  = Stats( args, args.stat_windowsize_nrsequences, 'stats_test' )   #stats over recent test data
    stats_train_alldata = Stats( args, args.nrinputfiles_train, 'stats_train' )   #stats over the most recent set of |trainingset| examples
    stats_test_alldata  = Stats( args, args.nrinputfiles_test, 'stats_test'   )   #stats over the most recent set of |testset| examples

    nrseq_inc = np.ceil( args.incremental_min_nrpoints / min( args.current_seq_length, dataloader_train.avgseqlength) ) 
    stats_train_inc = Stats( args, nrseq_inc, 'stats_train_inc' )  #stats over most recent incremental_min_nrpoints, for incremental methods

    random.seed( 100 * args.runnr ) 
    np.random.seed( 100 * args.runnr ) 
    tf.set_random_seed( 100 * args.runnr ) 
    print( 'runnr', args.runnr, 'after seed, rand:', random.random( ) , 'np rand', np.random.rand( ) ) 
    
    print( 'starting time: ', strftime( "%Y-%m-%d %H:%M:%S") ) 

    nrsequenceinputs = 4 #dx dy eos eod
    nrinputvars_network = nrsequenceinputs + args.nrauxinputvars;
    args.nroutputvars_raw = ( 2 + args.num_mixture * 6)  * args.useStrokeOutputVars + args.nrClassOutputVars 
    args.nroutputvars_final = ( 2 + 2)  * args.useStrokeOutputVars + args.nrClassOutputVars 

    print( "nrinputvars_network", nrinputvars_network) 
    print( "nrauxinputvars", args.nrauxinputvars) 
    print( "args.nroutputvars_final", args.nroutputvars_final) 
    
    trainpredictmode = "Predict"

    model = Model( args, trainpredictmode, False, nrinputvars_network, args.nroutputvars_raw, args.nrtargetvars, args.nrClassOutputVars, dataloader_train.rangemin, dataloader_train.rangelen, args.maxdigitlength_nrpoints ) 

    #store info from model in args so it's saved:
    args.seq_length = model.seq_length
        
    print( 'about to save config in', os.path.join( 'save/'+args.explabel+'r'+str( args.runnr) , 'config.pkl') ) 
    with open( os.path.join( 'save/'+args.explabel+'r'+str( args.runnr) , 'config.pkl') , 'wb')  as f:
        pickle.dump( args, f) 

    print_model( model ) 

    checkpoint_fullpath = ""
    nanfound = False
    nrnanbatches = 0
    train_on_output = 0

    printstate = args.reportstate
    
    with tf.Session( )  as sess:
        
        random.seed( 100 * args.runnr ) 
        np.random.seed( 100 * args.runnr ) 
        randop = tf.random_normal( [ 1 ], seed = random.random( ) ) #, seed = 1234
        print( 'runnr', args.runnr, 'after seed, rand:', random.random( ) , 'np rand', np.random.rand( ) , 'tf rand', sess.run( randop) ) 

        dataloader_train.createRandValues( )         
        dataloader_test.createRandValues( )         

        tf.initialize_all_variables( ) .run( ) 
        saver = tf.train.Saver( tf.trainable_variables( ) , max_to_keep = args.save_maxnrmodels_keep) 

        if args.useInitializers:
            init_op_weights = sess.run( model.init_op_weights) 

        if ( len( args.startingpoint) >0) :
          print( "Starting from saved model: ", args.startingpoint) 
          restoreModel( sess, args, model, dataloader_train, args.startingpoint) 

          if args.useInitializers:
            init_op_weights = sess.run( model.init_op_weights) 
          
        totnrbatches = 0
        totnrpoints = 0
        totnrpoints_trained = 0
        tstart = time.time( ) 
            
        nrbatches_per_epoch = dataloader_train.nrbatches_per_epoch
        epochnr = 0
        cont = True

        while cont: #batch loop
            
          if ( totnrbatches % nrbatches_per_epoch == 0 ) :
              learningrate_value = args.learning_rate * ( args.decay_rate ** epochnr) 
              learningrate = sess.run( model.learningrateop, feed_dict = {model.learningrate_ph: learningrate_value}) 
              epochnr += 1

          modes = [ "train" ]
          if totnrbatches > 0 and ( totnrbatches  % args.test_every_nrbatches == 0 ) :
              modes = [ "train", "test" ]
              
          for mode in modes:
              
                if mode == "train":
                    dataloader = dataloader_train
                    runseqlength = args.current_seq_length
                else:
                    dataloader = dataloader_test
                    runseqlength = model.seq_length                                

                if mode == "train":
                    totnrbatches += 1                
                    stats = stats_train
                    stats_alldata = stats_train_alldata
                    stats_inc = stats_train_inc
                else:
                    stats = stats_test
                    stats_alldata = stats_test_alldata
                    stats_inc = 0
                                        
                start_batch = time.time( ) 

                tstartdata = time.time( )       
                getbatch = True
                if epochnr > 0:
                  if args.train_on_own_output_method == 1 and mode == "train":
                    train_on_output = dataloader.getRandValue( )  < 1.0 / ( 2. + mean( model.batch_rmse_stroke, model.batch_rmse_class) )  #training --> rand ok
                    if train_on_output:
                      x = constructInputFromOutput( args, model, x, o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, o_eos, o_eod) 
                      getbatch = False

                if getbatch: #not training on own output
                    x, y, sequence_index = dataloader.next_batch( args, runseqlength )  #can contain multiple sequence
                tgetdata = ( time.time( )  - tstartdata)  / 60
                start_train = time.time( ) 

                batchsize_nrseq = len( x ) 
                
                zero_initial_state = sess.run( model.initial_state, feed_dict = {model.batch_size_ph: batchsize_nrseq, model.seq_length_ph: runseqlength})   #Get zero state given current batchsz
                
                if ( mode == "test")  or ( totnrbatches == 1)  or ( not args.usePreviousEndState) :
                    state = zero_initial_state
                else:
                    state = last_train_state
                                  
                feed = {model.input_data: x, model.target_data: y, model.initial_state: state, model.batch_size_ph: batchsize_nrseq, model.seq_length_ph: args.current_seq_length} #model.seq_length
                
                if args.useStrokeOutputVars and args.useStrokeLoss:
                    if mode == "train":
                        train_loss, last_train_state, lossvec, o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, o_eos, o_eod, o_classvars, o_classpred, regularization_term, loss_plain, lossnrpoints, maxabsweight, avgweight, _ = sess.run( [ model.loss_total, model.final_state, model.lossvector, model.pi, model.mu1, model.mu2, model.sigma1, model.sigma2, model.corr, model.eos, model.eod, model.classvars, model.classpred, model.regularization_term, model.loss_plain, model.lossnrpoints, model.maxabsweight, model.avgweight, model.train_op ], feed)  
                        state_report = last_train_state
                    else: #test --> omit train op, and don't replace state
                        train_loss, state_report, lossvec, o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, o_eos, o_eod, o_classvars, o_classpred, regularization_term, loss_plain, lossnrpoints, maxabsweight, avgweight = sess.run( [ model.loss_total, model.final_state, model.lossvector, model.pi, model.mu1, model.mu2, model.sigma1, model.sigma2, model.corr, model.eos, model.eod, model.classvars, model.classpred, model.regularization_term, model.loss_plain, model.lossnrpoints, model.maxabsweight, model.avgweight ], feed)  
                            
                    outputs = [ o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, o_eos, o_eod, o_classvars, o_classpred ]

                else: #no stroke loss, only learn classes
                    z = np.zeros( ( 1) , dtype = np.float32 ) 
                    [ o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, o_eos, o_eod ] = [ z, z, z, z, z, z, z, z ]
                    if mode == "train":
                        train_loss, last_train_state, output, lossvec, o_classvars, o_classpred, regularization_term, loss_plain, result4, result, result_before_mask, lossnrpoints, mask, classpred, targetdata_classvars, crossentropy, maxabsweight, avgweight,  _ = sess.run( [ model.loss_total, model.final_state, model.output, model.lossvector, model.classvars, model.classpred, model.regularization_term, model.loss_plain, model.result4, model.result, model.result_before_mask, model.lossnrpoints, model.mask, model.classpred, model.targetdata_classvars, model.crossentropy, model.maxabsweight, model.avgweight, model.train_op ], feed) 
                        state_report = last_train_state
                    else:
                        train_loss, state_report, output, lossvec, o_classvars, o_classpred, regularization_term, loss_plain, result4, result, mask,  maxabsweight, avgweight, classpred, targetdata_classvars  = sess.run( [ model.loss_total, model.final_state, model.output, model.lossvector, model.classvars, model.classpred, model.regularization_term, model.loss_plain, model.result4, model.result, model.mask, model.maxabsweight, model.avgweight, model.classpred, model.targetdata_classvars ], feed) 
                    outputs = [ o_classvars, o_classpred ]

                if mode == "train":
                    totnrpoints_trained += lossnrpoints
                
                weights_o = sess.run( model.outputWeight) ;

                nanfound = math.isnan( train_loss) 

                if nanfound:
                    print( ( "NAN encountered --> stopping.") ) 
                    sys.exit( ) ;

                end_train = time.time( ) 
                train_loss = train_loss.mean( ) 

                start_eval = time.time( ) 
                
                #evaluation
                if ( epochnr  % args.eval_every == 0 ) :
                  writefiles = totnrbatches % args.report_every == 0 and ( totnrbatches > 0) 
                  weights = 0
                  runtime = ( time.time( )  - tstart)  / 60                  
                  nanfound = nanfound or evaluate( sess, args, stats, stats_alldata, stats_inc, sequence_index, trainpredictmode, model, dataloader, outputdir, outputs, state_report, lossvec, train_loss, regularization_term, loss_plain, train_loss, weights,  nrinputvars_network, y, maxabsweight, avgweight, learningrate_value, train_on_output, epochnr, totnrbatches, totnrpoints_trained, writefiles, runtime, mode, printstate, batchsize_nrseq, x) 
                  stats.reset( ) 

                end_eval = time.time( ) 
                tot_time = end_train-start_train + end_eval-start_eval

                #saving:
                if ( not nanfound )  and ( mode == "train") :
                    if totnrbatches % args.save_every_nrbatches == 0 and ( totnrbatches > 0) :
                        savemodel( saver, sess, dataloader, args, totnrbatches) 

                if nanfound and mode == "train":
                    print( ( "NAN encountered --> stopping.") ) 
                    sys.exit( ) ;

                end_batch = time.time( ) 
                print ( "End of batch: time_train", end_train-start_train, "time ev", end_eval-start_eval, "tdata", tgetdata, "tot", tot_time, 'batch time', end_batch-start_batch, "sequences/sec", dataloader.nrseq_per_batch/tot_time) 

                if mode == "train":
                    if stats_train_inc.stats_stroke.totnrpoints >= args.incremental_min_nrpoints:

                        reached_threshold = False
                        if args.incremental_seq_length:
                            print( 'inc seq len: rmse', stats_train_inc.stats_stroke.rmse( ) , 'thr', args.threshold_rmse_stroke) 
                            if ( stats_train_inc.stats_stroke.rmse( )  < args.threshold_rmse_stroke)  and ( args.current_seq_length < model.seq_length) :
                                args.current_seq_length = min( model.seq_length, args.current_seq_length * 2 ) 
                                reached_threshold = True
                                print( "REACHED THRESHOLD! --> increasing cur_seq_length to ", args.current_seq_length, ' max ', model.seq_length) 
              
                        if args.incremental_nr_trainexamples:
                            print( 'inc nrtrainex: rmse stroke ', stats_train_inc.stats_stroke.rmse( ) , 'thr', args.threshold_rmse_stroke) 
                            if ( stats_train_inc.stats_stroke.rmse( )  < args.threshold_rmse_stroke)  and ( args.curnrtrainexamples < args.nrinputfiles_train ) :
                                args.curnrtrainexamples = min( 2 * args.curnrtrainexamples, args.nrinputfiles_train ) 
                                dataloader_train.curnrexamples = args.curnrtrainexamples
                                reached_threshold = True
                                print( "REACHED THRESHOLD! --> increasing curnrtrainexamples to ", args.curnrtrainexamples) 
                                dataloader.nrbatches_per_epoch = max( 1, int( args.curnrtrainexamples / dataloader.nrseq_per_batch) ) 
                                dataloader.reset_batch_pointer( args ) 
                                args.incremental_min_nrpoints = 50 * args.curnrtrainexamples
                                print ( "setting new nrbatches_per_epoch to: ", dataloader.nrbatches_per_epoch) 

                        if args.incremental_nr_digits:
                            print( 'inc nr digits: rmse ', stats_train_inc.stats_stroke.rmse( ) , ' thr', args.threshold_rmse_stroke) 
                            if ( stats_train_inc.stats_stroke.rmse( )  < args.threshold_rmse_stroke)  and ( args.curnrdigits < 10 )  :
                                args.curnrdigits = min( 2 * args.curnrdigits, 10 ) 
                                reached_threshold = True
                                print( "REACHED THRESHOLD! --> increasing curnrdigits to ", args.curnrdigits) 
                                dataloader.findAvailableExamples( args ) 
                                dataloader.nrbatches_per_epoch = max( 1, int( args.curnrtrainexamples / dataloader.nrseq_per_batch) ) 
                                dataloader.reset_batch_pointer( args ) 
                                print ( "setting new nrbatches_per_epoch to: ", dataloader.nrbatches_per_epoch) 

                        if reached_threshold: #reset rmse counters used for incremental learning
                            nrseq_inc = np.ceil( args.incremental_min_nrpoints / min( args.current_seq_length, dataloader_train.avgseqlength) ) 
                            stats_train_inc = Stats( args, nrseq_inc, 'stats_train_inc' )  #stats over most recent incremental_min_nrpoints, for incremental methods
                            
          #end of while loop ( batch) :
          cont = totnrpoints_trained <= args.maxnrpoints
          if ( stats_train.stats_stroke.rmse( )  < args.stopcrit_threshold_stroke_rmse_train ) :
              cont = False

        #end of run:
        print( 'End of run --> saving model' ) 
        savemodel( saver, sess, dataloader_train, args, totnrbatches) 
        print( 'done' ) 
        
if __name__ == '__main__':
  main( ) 


