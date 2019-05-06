# Fake-Face-Video-Detection

Source codes of Fake Face Video Detection Using Talking Profile

## Requirement

* Python 2.7.15
* matplotlib 2.2.2
* sklearn 0.20.3
* numpy 1.16.2
* imbalanced-learn 0.4.3

## Dataset

* [DeepfakeTIMIT](https://www.idiap.ch/dataset/deepfaketimit)  dateset.

  Download from [Google Drive](https://drive.google.com/file/d/1d-LfVOhr-M_-YlByF6iHZezayoDvJvfo/view?usp=sharing)

## Feature Extract

Use following tools to extract 3-D head rotation, 3-D head translation, mouth open-close magnitude (facial landmarks), eye gaze, eye blink state, and body pose.

1. [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace) (Baltrusaitis et al, 2018)
   
   It will generate csv file that contains face features for a video.
   
2. [LCR-NET](https://thoth.inrialpes.fr/src/LCR-Net/) (Rogez et al, 2019)
   
   I use InTheWild-ResNet50 model to extract body features. After downloading and building the repository, replace original demo.py with src/feature-extractor/body/demo.py and run the following command. The input is the directory of frame images of the video.
   ```shell
   python demo.py InTheWild-ResNet50 FRAMES_DIR 0
   ```

3. [In Ictu Oculi](https://github.com/danmohaha/WIFS2018_In_Ictu_Oculi) (Li et al, 2018)
   
   Use LRCN-VGG16 network to generate blink features. Replace original run_lrcn.py with src/feature-extractor/blink/run_lrcn.py and run.
   ```shell
    python run_lrcn.py \
    --input_vid_path=/path/to/video \
    --out_dir=where_to_save_output
   ```

### Naming format

All features extracted should be renamed as pidlabelvid_featuretype. Features of each subject should be in the seperate directory.

* pid: person id
* vid: video id
* label: f (fake) or r (real)
* featuretype: blink, body, or face
* For example, 001f00_blink indicates blink feature of fake video 00 of person 001. 

See sample features in sample_features.

## Train and Test

```shell
python main.py -i input-feature-path -o output-dir
```

* input-feature-path: path of extracted feature.
* output-dir: output directory.
