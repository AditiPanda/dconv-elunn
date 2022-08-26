# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 10:53:17 2017

@author: Aditi Panda
"""

import tensorflow as tf
import numpy as np
from glob import glob
from ops import *
from utils import *
from six.moves import xrange
import time
import os
import datetime


tf.reset_default_graph() 

class DnCNN(object):
    def __init__(self, sess, patch_size=40, batch_size=128,
                 output_size=40, input_c_dim=1, output_c_dim=1,
                 sigma=15, clip_b=0.025, lr=0.001, epoch=50,
                 ckpt_dir='./checkpoint-large-dataset', sample_dir='./sample-large-dataset',
                 test_save_dir='./test-large-dataset',
                 dataset='BSD400', testset='visual_gray', load_flag=True, initial_epoch=0): # test set changed on 12-11-17
#        tf.reset_default_graph() 
        self.sess = sess
        self.is_gray = (input_c_dim == 1)
        self.batch_size = batch_size
        self.patch_sioze = patch_size
        self.output_size = output_size
        self.input_c_dim = input_c_dim
        self.output_c_dim = output_c_dim
        self.sigma = sigma
        self.clip_b = clip_b
        self.lr = lr
        self.numEpoch = epoch
        self.ckpt_dir = ckpt_dir
        self.trainset = dataset
        self.testset = testset
        self.sample_dir = sample_dir
        self.test_save_dir = test_save_dir
        self.epoch = epoch
        self.save_every_epoch = 1
        self.eval_every_epoch = 1
        self.load_flag = load_flag
        self.initial_epoch = initial_epoch
        self.abs_epoch_num = self.initial_epoch
        # Adam setting (default setting)
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.alpha = 0.01
        self.epsilon = 1e-8        
        self.build_model()
        
    def build_model(self):
#        tf.reset_default_graph() 
        # input : [batchsize, patch_sioze, patch_sioze, channel]
        
        # the network structure has to be created in all cases (training for the first time, incremental training, and testing)
        self.create_variables() 
        
        # if load_flag = False, the model is being trained for the first time or tested. If the former is true, 
#        parameters like placeholders, learning algo and optimization functions need to be added to a collection, 
#        so that they could be easily extracted and used after restoration of the saved model.
        if not self.load_flag:
            print('block 1 of build_model')
   
#            self.create_variables()     
#            self.sess.run(self.init)
            print(self.initial_epoch)
            ################### commented on 11th Dec, 2017:  no need when create_vars is there; 
            #### also, it increases run time of subsequent epochs because of large .meta files
#            tf.add_to_collection('loss_op', self.loss)
##            print(tf.get_collection('loss_op'))
#            tf.add_to_collection('input', self.X)
#            tf.add_to_collection('target', self.X_)
#            tf.add_to_collection('output', self.Y_)
#            tf.add_to_collection('training_step', self.train_step)
            #########################
            print("[*] Created model successfully...")
        
        else: # this block is executed when incremental training is carried out. The value of the last executed epoch is found out,
        # and the initial epoch is set accordingly. The training now starts from this value of epoch.
            print('block 2 of build_model')


            model_dir = "%s-%s-%s" % (self.trainset,
                                      self.batch_size, self.patch_sioze)
            checkpoint_dir = os.path.join(self.ckpt_dir, model_dir)
            curr_path = os.getcwd()
            os.chdir(checkpoint_dir)
            
            # Find last executed epoch
            history = list(map(lambda x: int(x.split('-')[1][:-5]), glob('DnCNN.model-*.meta')))
            last_epoch = np.max(history)
            # Instantiate saver object using previously saved meta-graph
#            self.saver = tf.train.import_meta_graph('DnCNN.model-{}.meta'.format(last_epoch)) # commented on 11th Dec, 2017
            
            # find out latest version amongst saved models
            self.initial_epoch = last_epoch + 1
            self.abs_epoch_num = self.initial_epoch
            print(self.initial_epoch)
            os.chdir(curr_path)


    def create_variables(self):
# this function creates the network structure, i.e., the layers, the loss function, the optimization algos etc.
        self.X = tf.placeholder(tf.float32, [None, self.patch_sioze, self.patch_sioze, self.input_c_dim],
                                    name='noisy_image')
        self.X_ = tf.placeholder(tf.float32, [None, self.patch_sioze, self.patch_sioze, self.input_c_dim],
                                     name='clean_image')
        # layer 1
        with tf.variable_scope('conv1'):
            layer_1_output = self.layer(self.X, [3, 3, self.input_c_dim, 64], useBN=False)
            # layer 2 to 16
        with tf.variable_scope('conv2'):
            layer_2_output = self.layer(layer_1_output, [3, 3, 64, 64], d_rate=2)
#        print('conv2')
        with tf.variable_scope('conv3'):
            layer_3_output = self.layer(layer_2_output, [3, 3, 64, 64], d_rate=2)
        with tf.variable_scope('conv4'):
            layer_4_output = self.layer(layer_3_output, [3, 3, 64, 64], d_rate=2)
        with tf.variable_scope('conv5'):
            layer_5_output = self.layer(layer_4_output, [3, 3, 64, 64], d_rate=2)
        with tf.variable_scope('conv6'):
            layer_6_output = self.layer(layer_5_output, [3, 3, 64, 64], d_rate=2)
        with tf.variable_scope('conv7'):
            layer_7_output = self.layer(layer_6_output, [3, 3, 64, 64], d_rate=2)
        with tf.variable_scope('conv8'):
            layer_8_output = self.layer(layer_7_output, [3, 3, 64, 64], d_rate=2)
        with tf.variable_scope('conv9'):
            layer_9_output = self.layer(layer_8_output, [3, 3, 64, 64], d_rate=2)
#        with tf.variable_scope('conv10'):
#            layer_10_output = self.layer(layer_9_output, [3, 3, 64, 64], d_rate=2)
#        with tf.variable_scope('conv11'):
#            layer_11_output = self.layer(layer_10_output, [3, 3, 64, 64], d_rate=2)
#        with tf.variable_scope('conv12'):
#            layer_12_output = self.layer(layer_11_output, [3, 3, 64, 64], d_rate=2)
#        with tf.variable_scope('conv13'):
#            layer_13_output = self.layer(layer_12_output, [3, 3, 64, 64], d_rate=2)
#        with tf.variable_scope('conv14'):
#            layer_14_output = self.layer(layer_13_output, [3, 3, 64, 64], d_rate=2)
#        with tf.variable_scope('conv15'):
#            layer_15_output = self.layer(layer_14_output, [3, 3, 64, 64], d_rate=2)
#        with tf.variable_scope('conv16'):
#            layer_16_output = self.layer(layer_15_output, [3, 3, 64, 64], d_rate=2)
            # layer 17
        with tf.variable_scope('conv10'):
            self.Y = self.layer(layer_9_output, [3, 3, 64, self.output_c_dim], useBN=False, useELU=False)   
            
        # L2 loss
        self.Y_ = self.X - self.X_  # noisy image - clean image
        self.loss = (1.0 / self.batch_size) * tf.nn.l2_loss(self.Y_ - self.Y)
       
        optimizer = tf.train.AdamOptimizer(self.lr, name='AdamOptimizer')
        self.train_step = optimizer.minimize(self.loss)
        tf.summary.scalar('loss', self.loss)
        
        # create this init op after all variables specified, it helps in initializing all variables of the program (weights and biases)
        self.init = tf.global_variables_initializer()
        
        self.saver = tf.train.Saver(max_to_keep=51) # this will be used for saving and restoring trained models in binary files, i.e., checkpointing
        #        max_to_keep added on 11th Dec, 2017
        
        print('variables created')

    def conv_layer(self, inputdata, weightshape, b_init, stridemode, d_rate):
        # weights
        W = tf.get_variable('weights', weightshape,
                            initializer=tf.constant_initializer(get_conv_weights(weightshape, self.sess)))
#        print(W.shape)
        b = tf.get_variable('biases', [1, weightshape[-1]], initializer=tf.constant_initializer(b_init))
        # convolutional layer
#        print(d_rate)
        if d_rate == 1:
            return tf.add(tf.nn.conv2d(inputdata, W, strides=stridemode, padding="SAME"), b)  # SAME with zero padding
        else:
            return tf.add(tf.nn.atrous_conv2d(inputdata, W, rate=d_rate, padding="SAME"), b)  # SAME with zero padding
            
   
    def bn_layer(self, logits, output_dim, b_init=0.0):        
        alpha = tf.get_variable('bn_alpha', [1, output_dim], initializer=\
                                tf.constant_initializer(get_bn_weights([1, output_dim], self.clip_b, self.sess)))
        beta = tf.get_variable('bn_beta', [1, output_dim], initializer=\
                               tf.constant_initializer(b_init))
        return batch_normalization(logits, alpha, beta, isCovNet=True)
    
    def layer(self, inputdata, filter_shape, b_init=0.0, stridemode=[1, 1, 1, 1], useBN=True, useELU=True, d_rate=1):
#        print(filter_shape)
        logits = self.conv_layer(inputdata, filter_shape, b_init, stridemode, d_rate)
#        # this if-else added on 12-11-17, the 4 lines commented after this were there before
#        if useReLU == False:
#            output = logits
#        else:
#            if useBN:
#                output = tf.nn.relu(self.bn_layer(logits, filter_shape[-1]))
#            else:
#                output = tf.nn.relu(logits)
        if useELU:
            logits = tf.nn.elu(logits)            
#            logits = self.conv_layer(inputdata, [1, 1, 64, 64], b_init, stridemode, d_rate=1)
        if useBN:
            W_conv1 = tf.get_variable('weights_conv1', [1, 1, 64, 64],
                            initializer=tf.constant_initializer(get_conv_weights([1, 1, 64, 64], self.sess)))
            logits = tf.nn.conv2d(logits, W_conv1, strides=stridemode, padding="SAME")
            output = self.bn_layer(logits, filter_shape[-1])
        else:
            output = logits
        return output
    
    def train(self): 
        self.sess.run(self.init) # initialize the variables of the program, this has to be done in all cases i.e., 
#        training for the first time, incremental training, and testing
        
        if self.load_flag:              
#         load the latest trained model saved
            if self.load(self.ckpt_dir):
                print(" [*] Load SUCCESS (in train)")
            else:
                print(" [!] Load failed...(in train)")
            # extract variables saved in collections earlier in build_model function
            ##########commented on 11th Dec, 2017
#            self.train_step = tf.get_collection('training_step')[0]
#            self.X = tf.get_collection('input')[0]
#            self.X_ = tf.get_collection('target')[0]
#            self.Y_ = tf.get_collection('output')[0]
#            self.loss = tf.get_collection('loss_op')[0]
            
        # get data
        test_files = glob('./data/test/{}/*.png'.format(self.testset))
        test_data = load_images(test_files)  # list of array of different size, 4-D, pixel value range is 0-255
        data = load_data(filepath='./data/img_clean_pats.npy')
        numBatch = int(data.shape[0] / self.batch_size)
              
        # create file name and an empty list
        file_part1 = 'training-loss-'
        ext = '.npy'
        
       
        
        print("[*] Start training : ")
        print(datetime.datetime.now())
        start_time = time.time()
        for epoch in range(self.initial_epoch, self.epoch):
             # a list for storing loss values epoch wise
            loss_list = []
            for batch_id in xrange(numBatch):
                batch_images = data[batch_id * self.batch_size:(batch_id + 1) * self.batch_size, :, :, :]
                batch_images = np.array(batch_images / 255.0, dtype=np.float32)     #normalize the data to 0-1, line added for 12-11-17
#                print(batch_images.shape)
                train_images = add_noise(batch_images, self.sigma, self.sess)
#                print(train_images.shape)
#                _, loss, summary = self.sess.run([self.train_step, self.loss, merged], \
#                                                 feed_dict={self.X: train_images, self.X_: batch_images})
                _, loss = self.sess.run([self.train_step, self.loss],\
                                        feed_dict={self.X: train_images, self.X_: batch_images}) 
                loss_list.append(loss)                                        
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.6f" \
                      % (epoch, batch_id + 1, numBatch,
                         time.time() - start_time, loss))
            self.save(epoch)
            file_name = file_part1 + str(epoch) + ext                                      
            np.save(file_name, loss_list) 
#            self.evaluate(epoch, test_data)  # test_data value range is 0-255             
        print("[*] Finish training.")
        print(datetime.datetime.now())
        
            
    def save(self, epoch):
        # create the name of the folder containing the checkpoints
        model_name = "DnCNN.model"
        model_dir = "%s-%s-%s" % (self.trainset,
                                  self.batch_size, self.patch_sioze)
        checkpoint_dir = os.path.join(self.ckpt_dir, model_dir)
        
        # make the folder if it doesn't already exist
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        
        #save using the saver object created earlier
        print("[*] Saving model...")
        self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=epoch)
        
    def sampler(self, image):
        # set reuse flag to True
        # tf.get_variable_scope().reuse_variables()
        self.X_test = tf.placeholder(tf.float32, image.shape, name='noisy_image_test')
        # layer 1 (adpat to the input image)
        with tf.variable_scope('conv1', reuse=True):
            layer_1_output = self.layer(self.X_test, [3, 3, self.input_c_dim, 64], useBN=False)
        # layer 2 to 16
        with tf.variable_scope('conv2', reuse=True):
            layer_2_output = self.layer(layer_1_output, [3, 3, 64, 64], d_rate=2)
        with tf.variable_scope('conv3', reuse=True):
            layer_3_output = self.layer(layer_2_output, [3, 3, 64, 64], d_rate=2)
        with tf.variable_scope('conv4', reuse=True):
            layer_4_output = self.layer(layer_3_output, [3, 3, 64, 64], d_rate=2)
        with tf.variable_scope('conv5', reuse=True):
            layer_5_output = self.layer(layer_4_output, [3, 3, 64, 64], d_rate=2)
        with tf.variable_scope('conv6', reuse=True):
            layer_6_output = self.layer(layer_5_output, [3, 3, 64, 64], d_rate=2)
        with tf.variable_scope('conv7', reuse=True):
            layer_7_output = self.layer(layer_6_output, [3, 3, 64, 64], d_rate=2)
        with tf.variable_scope('conv8', reuse=True):
            layer_8_output = self.layer(layer_7_output, [3, 3, 64, 64], d_rate=2)
        with tf.variable_scope('conv9', reuse=True):
            layer_9_output = self.layer(layer_8_output, [3, 3, 64, 64], d_rate=2)
#        with tf.variable_scope('conv10', reuse=True):
#            layer_10_output = self.layer(layer_9_output, [3, 3, 64, 64], d_rate=2)
#        with tf.variable_scope('conv11', reuse=True):
#            layer_11_output = self.layer(layer_10_output, [3, 3, 64, 64], d_rate=2)
#        with tf.variable_scope('conv12', reuse=True):
#            layer_12_output = self.layer(layer_11_output, [3, 3, 64, 64], d_rate=2)
#        with tf.variable_scope('conv13', reuse=True):
#            layer_13_output = self.layer(layer_12_output, [3, 3, 64, 64], d_rate=2)
#        with tf.variable_scope('conv14', reuse=True):
#            layer_14_output = self.layer(layer_13_output, [3, 3, 64, 64], d_rate=2)
#        with tf.variable_scope('conv15', reuse=True):
#            layer_15_output = self.layer(layer_14_output, [3, 3, 64, 64], d_rate=2)
#        with tf.variable_scope('conv16', reuse=True):
#            layer_16_output = self.layer(layer_15_output, [3, 3, 64, 64], d_rate=2)
        # layer 17
        with tf.variable_scope('conv10', reuse=True):
            self.Y_test = self.layer(layer_9_output, [3, 3, 64, self.output_c_dim], useBN=False, useELU=False)
    
    def load(self, checkpoint_dir):
        '''Load checkpoint file'''
        print("[*] Reading checkpoint...")
        # create the name of the folder containing the checkpoints
        model_dir = "%s-%s-%s" % (self.trainset, self.batch_size, self.patch_sioze)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        
        
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False
    
    def forward(self, noisy_image):
        # assert noisy_image is range 0-1
        self.sampler(noisy_image)
        return self.sess.run(self.Y_test, feed_dict={self.X_test: noisy_image})
    
    def test(self):
        """Test DnCNN"""
        # init variables
        self.sess.run(self.init)        
        
        print (self.test_save_dir)
        test_files = glob('./data/test/{}/*.png'.format(self.testset))
        print(len(test_files))
        
        # three lines commented on 12-11-17
#        # load testing input
#        print("[*] Loading test images ...")
#        test_data = load_images(test_files)  # list of array of different size, range 0-255

        if self.load(self.ckpt_dir):
            print(" [*] Load SUCCESS (in test)")
        else:
            print(" [!] Load failed...(in test)")
        psnr_sum = 0
        print("[*] " + 'noise level: ' + str(self.sigma) +  " start testing...") # added on 12-11-17
        print(datetime.datetime.now())
        for idx in xrange(len(test_files)):
            print(idx)            
            # noisy_image = add_noise(test_data[idx] / 255.0, self.sigma, self.sess)  # ndarray, commented on 12-11-17
            
            # two lines added on 12-11-17
            test_data = load_image(test_files[idx])
            noisy_image = add_noise(test_data/ 255.0, self.sigma, self.sess)  # ndarray
            
            predicted_noise = self.forward(noisy_image)
            
            # two lines commented on 12-11-17
            # output_clean_image = noisy_image - predicted_noise          
            # groundtruth = np.clip(test_data[idx], 0, 255).astype('uint8') 
           
            # two lines added on 12-11-17
            output_clean_image = noisy_image - predicted_noise          
            groundtruth = np.clip(test_data, 0, 255).astype('uint8')
            
            noisyimage = np.clip(255 * noisy_image, 0, 255).astype('uint8')
            outputimage = np.clip(255 * output_clean_image, 0, 255).astype('uint8')
            
            # calculate PSNR
            psnr = cal_psnr(groundtruth, outputimage)
            print(psnr) # added on 12-11-17
            
            psnr_sum += psnr
            
           # save_images(groundtruth, noisyimage, outputimage, os.path.join(self.test_save_dir, 'test%d.png' % idx)) # commented on 12-11-17
           
           # two lines added on 12-11-17
            save_image(noisyimage, os.path.join(self.test_save_dir, 'noisy%d.png' % idx))
            save_image(outputimage, os.path.join(self.test_save_dir, 'denoised%d.png' % idx))
           
        avg_psnr = psnr_sum / len(test_files)
        avg_psnr = psnr_sum / len(test_files)
        print("--- Average PSNR %.2f ---" % avg_psnr)
        print(datetime.datetime.now())
    

    def evaluate(self, epoch, test_data):
               
        print("[*] Evaluating...")
        psnr_sum = 0
        print(datetime.datetime.now())
        for idx in xrange(len(test_data)):
            # find out the max gray value in the current test image
            print (np.max(test_data[idx]))
            assert np.max(test_data[idx]) > 1
                         
            noisy_image = add_noise(test_data[idx] / 255.0, self.sigma, self.sess)  # ndarray
            predicted_noise = self.forward(noisy_image)
            output_clean_image = noisy_image - predicted_noise
            
            groundtruth = np.clip(test_data[idx], 0, 255).astype('uint8')
            noisyimage = np.clip(255 * noisy_image, 0, 255).astype('uint8')
            outputimage = np.clip(255 * output_clean_image, 0, 255).astype('uint8')
            
            # calculate PSNR
            psnr = cal_psnr(groundtruth, outputimage)
            psnr_sum += psnr
            save_images(groundtruth, noisyimage, outputimage,
                        os.path.join(self.sample_dir, 'test%d_%d.png' % (idx, epoch)))
        avg_psnr = psnr_sum / len(test_data)
        
        file_part1 = 'avg-psnr-eval-'
        ext = '.npy'
        file_name = file_part1 + str(epoch) + ext
        np.save(file_name, avg_psnr)          
                  
        print("--- Test ---- Average PSNR %.2f ---" % avg_psnr)
        print(datetime.datetime.now())

        
    
            
            
            