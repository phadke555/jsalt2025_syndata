# Notes

finetuning on AMI headset train data (ihm) for a single speaker at a time

using python 3.10

conda environment

had to make a few changes to the original F5-TTS repo to get things working. changes made to files model.dataset.load_dataset, model.trainer.save_checkpoint, model.trainer.load_checkpoint. had to adjust the way the optimizer state dict was extracted and loaded with the accelerate version. this was an important change to get the code running. for the load_dataset function, there was an inherent bug in the branch of the tree for CustomDatasetPath where preprocessed_mel was referenced before assignment so i patched that by editing the local file.



Expecting data to be prepared with the following format. Folder with wavs. Metadata csv with the headers, audio_path, text, duration.
Duration json which is basically a dictionary of duration to a list of the lengths of all of the files.
And then a raw.arrow file. 
This is assuming single speaker finetuning.