"""
Audio-Visual ASD Model Trainer
------------------------------
This script trains and evaluates an **Audio-Visual Active Speaker Detection (ASD) model** 
using deep learning. It processes **audio and visual data** from a dataset and applies 
a multimodal approach to improve speaker detection accuracy.

Main Functions:
1. **Argument Parsing (`parser()`)**:
   - Defines hyperparameters such as learning rate (`--lr`), batch size (`--batchSize`), 
     number of epochs (`--maxEpoch`), and dataset paths (`--datasetPath`).
   - Supports both **training** and **evaluation** modes (`--evaluation` flag).

2. **Data Loading (`main(args)`)**:
   - Uses `train_loader` for loading training data and `val_loader` for validation/testing.
   - Creates PyTorch DataLoaders for efficient **batch processing**.

3. **Model Training & Evaluation**:
   - Loads a pre-trained model if available (`model_XXXX.model`).
   - Trains the model for `maxEpoch` epochs.
   - Evaluates the model at test intervals and logs performance.
   - Saves the best-performing model based on **Mean Average Precision (mAP)**.

4. **Model Saving & Loading**:
   - **Saves parameters** after training (`model_XXXX.model`).
   - **Loads parameters** for resuming training or evaluation (`--eval_model_path`).

Usage:
- **Training**: `python train.py` (with default arguments)
- **Evaluation**: `python train.py --evaluation --eval_model_path <model_path>`

Requirements:
- Dataset stored at `--datasetPath` (default: `/mnt/data/datasets/AVDIAR_ASD/`).
- Model outputs saved to `--savePath`.
"""

import os, glob, time
import argparse
from model import *
from dataLoader_Image_audio import train_loader, val_loader

def parser():

    args = argparse.ArgumentParser(description="ASD Trainer")

    args.add_argument("--lr", type=float, default=0.0001, help="Learning Rate")
    args.add_argument('--lrDecay', type=float, default=0.95, help='Learning rate decay rate')
    args.add_argument('--maxEpoch', type=int, default=25, help='Maximum number of epochs')
    args.add_argument('--testInterval', type=int, default=1, help='Test and save every [testInterval] epochs')
    args.add_argument('--batchSize', type=int, default=32, help='Dynamic batch size, default is 500 frames.')
    args.add_argument('--nDataLoaderThread', type=int, default=4, help='Number of loader threads')
    args.add_argument('--datasetPath', type=str, default="/mnt/data/datasets/AVDIAR_ASD/", help='Path to the ASD Dataset')
    args.add_argument('--loadAudioSeconds', type=float, default=3, help='Number of seconds of audio to load for each training sample')
    args.add_argument('--loadNumImages', type=int, default=1, help='Number of images to load for each training sample')
    args.add_argument('--savePath', type=str, default="exps/exp1")
    args.add_argument('--evalDataType', type=str, default="val", help='The dataset for evaluation, val or test')
    args.add_argument('--evaluation', dest='evaluation', action='store_true', help='Only do evaluation')
    args.add_argument('--eval_model_path', type=str, default="path not specified", help="model path for evaluation")

    args = args.parse_args()

    return args

def main(args):

    loader = train_loader(trialFileName = os.path.join(args.datasetPath, 'csv/train_loader.csv'), \
                          audioPath      = os.path.join(args.datasetPath , 'clips_audios/'), \
                          visualPath     = os.path.join(args.datasetPath, 'clips_videos/train'), \
                          **vars(args))
    trainLoader = torch.utils.data.DataLoader(loader, batch_size = args.batchSize, shuffle = True, num_workers = args.nDataLoaderThread)

    loader = val_loader(trialFileName = os.path.join(args.datasetPath, 'csv/val_loader.csv'), \
                        audioPath     = os.path.join(args.datasetPath , 'clips_audios'), \
                        visualPath    = os.path.join(args.datasetPath, 'clips_videos', args.evalDataType), \
                        **vars(args))
    valLoader = torch.utils.data.DataLoader(loader, batch_size = args.batchSize, shuffle = False, num_workers = 16)
    
    if args.evaluation == True:
        s = model(**vars(args))

        if args.eval_model_path=="path not specified":
            print('Evaluation model parameters path has not been specified')
            quit()
        
        s.loadParameters(args.eval_model_path)
        print("Parameters loaded from path ", args.eval_model_path)
        mAP = s.evaluate_network(loader = valLoader, **vars(args))
        print("mAP %2.2f%%"%(mAP))
        quit()    
    
    args.modelSavePath = os.path.join(args.savePath, 'model')
    os.makedirs(args.modelSavePath, exist_ok=True)
    args.scoreSavePath    = os.path.join(args.savePath, 'score.txt')
    modelfiles = glob.glob('%s/model_0*.model'%args.modelSavePath)
    modelfiles.sort()  
    if len(modelfiles) >= 1:
        print("Model %s loaded from previous state!"%modelfiles[-1])
        epoch = int(os.path.splitext(os.path.basename(modelfiles[-1]))[0][6:]) + 1
        s = model(epoch = epoch, **vars(args))
        s.loadParameters(modelfiles[-1])
    else:
        epoch = 1
        s = model(epoch = epoch, **vars(args))

    mAPs = []
    scoreFile = open(args.scoreSavePath, "a+")
    bestmAP = 0
    while(1):        
        loss, lr = s.train_network(epoch = epoch, loader = trainLoader, **vars(args))
        
        if epoch % args.testInterval == 0:        
            
            mAPs.append(s.evaluate_network(epoch = epoch, loader = valLoader, **vars(args)))
            if mAPs[-1] > bestmAP:
                bestmAP = mAPs[-1]
                s.saveParameters(args.modelSavePath + "/best.model")
                torch.save(s.state_dict(), os.path.join(args.modelSavePath, "final_model.pth"))  # Save the model here
            print(time.strftime("%Y-%m-%d %H:%M:%S"), "%d epoch, mAP %2.2f%%, bestmAP %2.2f%%"%(epoch, mAPs[-1], max(mAPs)))
            scoreFile.write("%d epoch, LR %f, LOSS %f, mAP %2.2f%%, bestmAP %2.2f%%\n"%(epoch, lr, loss, mAPs[-1], max(mAPs)))
            scoreFile.flush()

        if epoch >= args.maxEpoch:
            quit()

        epoch += 1

if __name__=="__main__":

    args = parser()

    main(args)