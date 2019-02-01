import os
import glob
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
from keras.models import Sequential, load_model
import numpy as np
model_path = "TrainingCheckpoints/DO_trained.hdf5"
model=load_model(model_path)
import ROOT as r
import re
from shutil import copyfile
from root_numpy import root2array, array2tree, array2root, list_trees, list_directories
#from fourvector import FourMomentum

rootfiledir = "/storage/b/akhmet/merged_files_from_naf/04_09_2018_HToTauTau_with_SVFit_v2_postsync/"
newrootfiledir = "/storage/b/akhmet/merged_files_from_naf/04_09_2018_HToTauTau_with_SVFit_v2_postsync_summerstudentDNN/"

branches = [
    "pt_1",
    "eta_1",
    "phi_1",
    "e_1",
    "m_1",
    "pt_2",
    "eta_2",
    "phi_2",
    "e_2",
    "m_2",
    "met",
    "metphi",
    "genbosonmass",
]

def get_files_information(l, matchmattern=r'.*'):
    #lmatched = [f for f in l if re.search(matchmattern, f) if "M80_" not in f and "M90_" not in f]
    lmatched = [f for f in l if re.search(matchmattern, f)]
    new_filenames = [ f.strip().split("_")[0].replace("ToTauTauM","") for f in lmatched]
    masses = [int(re.findall(r'\d+', f)[0]) for f in new_filenames]
    binnings = [ 50 if m < 150 else 100 for m in masses]
    return zip(lmatched, new_filenames, masses, binnings)

def sorted_nicely(l):
    """ Sort the given iterable in the way that humans expect: alphanumeric sort (in bash, that's 'sort -V')"""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)

inputsamples_path = os.path.join(rootfiledir,"*/*.root")
filepaths = sorted_nicely(glob.glob(inputsamples_path))
#filepaths = [ p for p in filepaths if "HToTauTau" in p]
filenames = [os.path.basename(p) for p in filepaths]
info = get_files_information(filenames)

for f,p,inf in zip(filenames,filepaths,info):
    nick = f.replace(".root","")
    if "M80_" in nick or "M90_" in nick:
        continue
    treename = "ntuple"
    output_file = p.replace(rootfiledir,newrootfiledir)
    if not os.path.exists(os.path.dirname(output_file)):
        print "creating new directory:",os.path.dirname(output_file)
        os.makedirs(os.path.dirname(output_file))
    print "copying file from",p,"to",output_file
    copyfile(p, output_file)
    F = r.TFile(output_file, "UPDATE")

    for ch in ["mt","et","tt","em"]:
        foldername = "%s_nominal"%ch
        array = root2array(p,"/".join([foldername,treename]),branches = branches)
        if len(array) > 2:
            inputarray = []
            masseslist = []
            outputs = {}
            outputs_corrected = {}

            for i,line in enumerate(array):
                if ch in ["mt","et"]:
                    inputarray.append((0,1,0,line[0], line[1], line[2], line[3],line[4], line[5], line[6], line[7], line[8], line[9], line[10], line[11]))
                elif ch in ["tt"]:
                    inputarray.append((0,0,1,line[0], line[1], line[2], line[3],line[4], line[5], line[6], line[7], line[8], line[9], line[10], line[11]))
                elif ch in ["em"]:
                    inputarray.append((1,0,0,line[0], line[1], line[2], line[3],line[4], line[5], line[6], line[7], line[8], line[9], line[10], line[11]))
                masseslist.append(line[12])
            masses = np.array(masseslist)
            outputs[nick] = model.predict(inputarray).flatten()
            outputs_corrected[nick] = ((outputs[nick] - 115.)/0.4)
            diff = (outputs[nick]-masses)/masses
            diffcorr = (outputs_corrected[nick]-masses)/masses
            print nick, "UNCORRECTED", "median, down, up:",np.percentile(diff,50.0),np.percentile(diff,15.9), np.percentile(diff,84.1), "absolute median, down, up:", np.percentile(outputs[nick],50.0),np.percentile(outputs[nick],15.9), np.percentile(outputs[nick],84.1)
            print nick, "CORRECTED", "median, down, up:",np.percentile(diffcorr,50.0),np.percentile(diffcorr,15.9), np.percentile(diffcorr,84.1), "absolute median, down, up:", np.percentile(outputs_corrected[nick],50.0),np.percentile(outputs_corrected[nick],15.9), np.percentile(outputs_corrected[nick],84.1)
            print "-----------------------------------------"
            outputs[nick] = np.array(outputs[nick], dtype=[('m_DDT',np.float32)])
            outputs_corrected[nick] = np.array(outputs_corrected[nick], dtype=[('m_DDTcorr',np.float32)])


            if not foldername in list_directories(output_file):
                F.mkdir(foldername)
            tree = F.Get("/".join([foldername,treename]))
            getattr(F, foldername).cd()
            print "writing DNN outputs to tree"
            tree = array2tree(outputs[nick], name = treename, tree = tree)
            tree = array2tree(outputs_corrected[nick], name = treename, tree = tree)
            print "Saving tree to file"
            F.Write("",r.TObject.kOverwrite)
    print "Closing file"
    F.Close()

exit()

inputs = []
#input labels:
# first 3 values determine decay type
# (1,0,0) both leptoinc
# (0,1,0) mixed
# (0,0,1) both hadronic
# then
# p1pt, p1eta, p1phi, p1E, p1m
# p2pt, p2wta, p2phi, p2E, p2m
# MET
# phi_MET

# example inputs for 125 GeV Higgs decay
line = (0,1,0,70.699005127,0.943030714989,-1.73818576336,12.432457617,0.000510999991093,28.8855628967,0.000565979687963,0.906305015087,5.60208343102,1.58043241501,37.1358261108,-1.37983822823)
inputs.append(line)
inputs = np.array(inputs) 

# feed inputs to model   
model_prediction = model.predict(inputs)

# adjust model overestimation
#adjustment for overestimation
#for i in xrange(len(model_prediction)):
#    model_prediction[i] -= 115 - 0.60 * 125

# print estimated mass
print model_prediction
