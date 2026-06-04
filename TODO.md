TODO:

1. Simple models:
- (Stacked) LSTM
- RNN
- GRU
Score 25% on accuracy on Intra, since they:
- Guess one type all of the time, so it will guarantee 25% score.
However each run or model type it differs what has been choosen, but confusion matrix is:
0 1 0 0
0 1 0 0
0 1 0 0
0 1 0 0
This may differ where column of 1 is.
EEGNet scores 58.5% CNN1D scores 74.9% and CNN+Transformers scores 68.0%

2. Simple and more complex ones:
- (Stacked) LSTM
- RNN
- GRU
- CNN1D + Transformers
- EEGNet
Score 25% on accuracy on Cross, since they:
- Guess one type all of the time, so it will guarantee 25% score.
However each run or model type it differs what has been choosen, but confusion matrix is:
0 1 0 0
0 1 0 0
0 1 0 0
0 1 0 0
This may differ where column of 1 is.
The only model that scores above 25% is CNN1D with 43.9%.

Example run of RNN:
Intra
--Device       : cuda
--Model        : rnn  (198,020 parameters)
--Experiment   : intra
--Epochs (max) : 50 with Patience: 10
  Epoch   1/50  - train loss 1.6718  acc 0.240  - val loss 1.4383  acc 0.258  - lr 9.99e-04
  Epoch   5/50  - train loss 1.6218  acc 0.236  - val loss 1.4223  acc 0.258  - lr 9.76e-04
  Epoch  10/50  - train loss 1.5569  acc 0.242  - val loss 1.4200  acc 0.258  - lr 9.05e-04
  Epoch  15/50  - train loss 1.5188  acc 0.236  - val loss 1.4252  acc 0.258  - lr 7.94e-04
  Epoch  20/50  - train loss 1.4967  acc 0.236  - val loss 1.4059  acc 0.258  - lr 6.55e-04

--Warning--                     
Early stopping at epoch 24 (no improvement for 10 epochs)

              precision    recall  f1-score   support

        Rest      0.000     0.000     0.000        68
       Motor      0.000     0.000     0.000        68
  Math/Story      0.000     0.000     0.000        68
Working Mem.      0.250     1.000     0.400        68

    accuracy                          0.250       272
   macro avg      0.062     0.250     0.100       272
weighted avg      0.062     0.250     0.100       272

  Overall Accuracy : 0.250  (25.0%)
  Macro F1         : 0.100

Cross
  Device       : cuda
  Model        : rnn  (198,020 parameters)
  Experiment   : cross (chunked, 8 chunks)
Epoch   1/50  - train loss 0.8768  acc 0.819  - val loss 4.3588  acc 0.000
Epoch   5/50  - train loss 1.1506  acc 0.692  - val loss 3.6869  acc 0.000
Epoch  10/50  - train loss 3.1996  acc 0.083  - val loss 4.1259  acc 0.000
Epoch  15/50  - train loss 2.6943  acc 0.062  - val loss 3.4657  acc 0.000

--Warning--                     
Early stopping at epoch 17 (no improvement for 10 epochs)
              precision    recall  f1-score   support

        Rest      0.000     0.000     0.000       408
       Motor      0.000     0.000     0.000       408
  Math/Story      0.250     1.000     0.400       408
Working Mem.      0.000     0.000     0.000       408

    accuracy                          0.250      1632
   macro avg      0.062     0.250     0.100      1632
weighted avg      0.062     0.250     0.100      1632

  Overall Accuracy : 0.250  (25.0%)
  Macro F1         : 0.100

Example run of CNN1D:
Intra
--Device       : cuda
--Model        : cnn1d  (336,900 parameters)
--Experiment   : intra
--Epochs (max) : 50 with Patience: 10
  Epoch   1/50  - train loss 1.1574  acc 0.549  - val loss 1.4475  acc 0.258  - lr 9.99e-04
  Epoch   5/50  - train loss 0.6326  acc 0.889  - val loss 0.6550  acc 0.894  - lr 9.76e-04
  Epoch  10/50  - train loss 0.4767  acc 0.959  - val loss 0.5061  acc 0.954  - lr 9.05e-04
  Epoch  15/50  - train loss 0.4143  acc 0.988  - val loss 0.4228  acc 1.000  - lr 7.94e-04
  Epoch  20/50  - train loss 0.3938  acc 1.000  - val loss 0.4073  acc 0.991  - lr 6.55e-04
  Epoch  25/50  - train loss 0.3793  acc 1.000  - val loss 0.3930  acc 1.000  - lr 5.00e-04
  Epoch  30/50  - train loss 0.3779  acc 1.000  - val loss 0.3918  acc 0.995  - lr 3.45e-04
  Epoch  35/50  - train loss 0.3727  acc 1.000  - val loss 0.3832  acc 1.000  - lr 2.06e-04
  Epoch  40/50  - train loss 0.3702  acc 1.000  - val loss 0.3824  acc 1.000  - lr 9.55e-05
  Epoch  45/50  - train loss 0.3703  acc 1.000  - val loss 0.3848  acc 1.000  - lr 2.45e-05
  Epoch  50/50  - train loss 0.3685  acc 1.000  - val loss 0.3844  acc 1.000  - lr 0.00e+00

              precision    recall  f1-score   support

        Rest      0.932     1.000     0.965        68
       Motor      0.705     0.809     0.753        68
  Math/Story      0.974     0.544     0.698        68
Working Mem.      0.663     0.809     0.728        68

    accuracy                          0.790       272
   macro avg      0.818     0.790     0.786       272
weighted avg      0.818     0.790     0.786       272

  Overall Accuracy : 0.790  (79.0%)
  Macro F1         : 0.786

Cross
              precision    recall  f1-score   support

        Rest      0.729     0.375     0.495       408
       Motor      0.159     0.071     0.098       408
  Math/Story      0.381     0.020     0.037       408
Working Mem.      0.293     0.875     0.439       408

    accuracy                          0.335      1632
   macro avg      0.390     0.335     0.267      1632
weighted avg      0.390     0.335     0.267      1632

  Overall Accuracy : 0.335  (33.5%)
  Macro F1         : 0.267

  Experiment  Model   Accuracy  Macro F1  Parameters  Last Epoch  Training Time (s)
  ----------  ------  --------  --------  ----------  ----------  -----------------
  Intra       CNN-1D  0.790     0.786     336900      50          546.781          
  Cross       CNN-1D  0.335     0.267     336900      23          248.399          

Both have relatively high loss and low accuracy.

I want you to study my project in depth.
If there are any bugs, wrong implementations etc. give these first and give suggestions on how to fix them. How come the runs are different from eachother while all seeds have been set?

Then I want you to work on my data preprocessing. Especially my data downsampling and normalisation could use some work. What is the best suited to do for this kind of task and how?

After data preprocessing:
I want you to to guide me through changes that will help my project.
If you want to improve a file, ask me first with what you want to change, how and what it will do/improve. Do this per file you can improve.
I dont want you to change multiple files at once for 1 implementation.
Also make two models yourself that would suit this problem that can be called similar to the other models in /models folder: one that is similar to CNN1D (since it scores the highest) and one different from the others.

Continue the previous task.

You are rewriting a multi-file Python project step-by-step. The required order is:
- loader
- main
- trainer
- stacked_lstm
- models (the two models inside)
STRICT RULES:
- Resume from where you left off (do NOT restart from loader if already completed)
- Output ONLY ONE FILE per response
- Do NOT include explanations, commentary, or summaries
- Do NOT repeat previous files
- If a file is already done, skip it automatically and move to the next
- Prefer PATCH/DIFF format (git-style unified diff). If not possible, output full file only for the current step
-Stop immediately after finishing the current file
GOAL:
- Minimize token usage and continue execution efficiently until all files are completed.

START NOW:
State briefly which file you are continuing with (1 line max), then output only the code/diff.