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
2. [LCR-NET](https://thoth.inrialpes.fr/src/LCR-Net/) (Rogez et al, 2019)
3. [In Ictu Oculi](https://github.com/danmohaha/WIFS2018_In_Ictu_Oculi) (Li et al, 2018)

### Naming format

All features should be named as pid_vid_featuretype. Features of each subject should be in the seperate directory.

* pid: person id
* vid: video id
* featuretype: blink, body, or face

## Train and Test

```shell
python main.py -i input-feature-path -o output-dir
```

* input-feature-path: path of extracted feature.
* output-dir: output directory.
