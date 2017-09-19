import json
import pandas as pd
import os
from os.path import join
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import shutil

class DDSMData:
    def __init__(self,root_path,label_json,image_path,mask_path='.',n_splits=1,split_seed='None'):
        self.root_path = root_path
        self.label_json = label_json
        self.image_path = image_path
        self.mask_path = mask_path
        self.img_names = self.load_image_names()
        self.msk_names = self.load_mask_names()
        self.val_names = self.get_valid_labels()
        
    def load_json_file(self,path):
        data = open(path, 'r').read()
        try:
            return json.loads(data)
        except ValueError, e:
            raise MalformedJsonFileError('%s when reading "%s"' % (str(e),path))
    
    def load_labels(self):
        f = self.load_json_file(join(self.root_path,self.label_json))
        return f
        
    def load_file_names(self,mypath):
        f = []
        for (dirpath, dirnames, filenames) in os.walk(mypath):
            f.extend(filenames)
            break
        return f
    
    def load_image_names(self):
        f = self.load_file_names(join(self.root_path,self.image_path))
        return f
    
    def load_mask_names(self):
        f = self.load_file_names(join(self.root_path,self.mask_path))
        return f
        
    def get_valid_labels(self):
        f = [a for a in self.img_names if a in self.msk_names]
        return f
    
    def create_ttv_shuffle_splits(self,n_splits,train_size=0.7,val_size=0.15,test_size=0.15,rand_seed=None):
        #http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html
        train_size_tot = train_size+val_size
        val_prop = float(val_size)/train_size_tot
        train_prop = float(train_size)/train_size_tot
        sss_trainval_test = StratifiedShuffleSplit(n_splits=n_splits, test_size=1.0-train_size_tot, \
                                                   train_size=train_size_tot, random_state=rand_seed)
        sss_train_val = StratifiedShuffleSplit(n_splits=1, test_size=val_prop,\
                                               train_size=train_prop, random_state=rand_seed)
        #getting labels
        lab_dict = self.load_labels()
        imgs = np.array(self.val_names)
        labs = np.array([lab_dict[a] for a in imgs])
        #getting test-trainval splits
        trainidxint = []
        testidx = []
        trainidx = []
        validx = []
        for trnvalx,tstx in sss_trainval_test.split(imgs,labs):
            trainidxint.append(trnvalx)
            testidx.append(tstx)
            for trnx, valx in sss_train_val.split(imgs[trnvalx],labs[trnvalx]): 
                    trainidx.append(trnx)
                    validx.append(valx)
        
        #creating final train, test, val sets
        x_train = []
        x_test = []
        x_val = []
        y_train = []
        y_test = []
        y_val= []
        x_trainval = []
        y_trainval = []
        for ii in range(len(trainidx)):
            x_trainval.append(imgs[trainidxint[ii]])
            y_trainval.append(labs[trainidxint[ii]])
            x_test.append(imgs[testidx[ii]])
            y_test.append(labs[testidx[ii]])
            x_train.append(x_trainval[-1][trainidx[ii]])
            y_train.append(y_trainval[-1][trainidx[ii]])
            x_val.append(x_trainval[-1][validx[ii]])
            y_val.append(y_trainval[-1][validx[ii]])
            
        return x_train, y_train, x_val, y_val, x_test, y_test
    
    def copy_file(self,source,dest):
        try:
            shutil.copyfile(source, dest)
        except ValueError:
            print("Error: File does not exist!")  
            
    def make_new_direc(self,name):
        if os.path.exists(name):
            shutil.rmtree(name)
            os.makedirs(name)
        else:
            os.makedirs(name)
    
    def write_splits(self,x_train,x_val,x_test,log_dir):
        """
        designed to emulate original file structure for ddsm tanda experiments -- can be improved
        """
        
        #defining and creating directory structure
        log_direc = join(self.root_path,log_dir)
        train_direc = join(self.root_path,'train_set')
        val_direc = join(self.root_path,'val_set')
        test_direc = join(self.root_path,'test_set')
        
        self.make_new_direc(log_direc)
        self.make_new_direc(train_direc)
        self.make_new_direc(val_direc)
        self.make_new_direc(test_direc)
            
        #writing log files
            
        with open(join(log_direc,"train.txt"), "w") as outfile:
            for s in x_train:
                outfile.write("%s\n" % s)
                
        with open(join(log_direc,"val.txt"), "w") as outfile:
            for s in x_val:
                outfile.write("%s\n" % s)
                
        with open(join(log_direc,"test.txt"), "w") as outfile:
            for s in x_test:
                outfile.write("%s\n" % s)
                
        #copying files to directory structure
        if not os.path.exists(train_direc):
            os.makedirs(train_direc)
        for file_name in x_train:
            full_file_name = join(self.root_path, self.image_path, file_name)
            new_file = join(train_direc,file_name)
            self.copy_file(full_file_name,new_file)
            
        for file_name in x_val:
            full_file_name = join(self.root_path, self.image_path, file_name)
            new_file = join(val_direc,file_name)
            self.copy_file(full_file_name,new_file)
            
        for file_name in x_test:
            full_file_name = join(self.root_path, self.image_path, file_name)
            new_file = join(test_direc,file_name)
            self.copy_file(full_file_name,new_file)
        
                
            