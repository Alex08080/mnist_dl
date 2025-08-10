## 🚀 Guide d’utilisation

### Installation  
pip install -r requirements.txt

### Lancement de l’entraînement  
python train.py --optimizer sgd --epochs 20 --save_model 1 --save_metrics 1 --plot_confusion 1 --lr 0.001

### Visualisation TensorBoard  
tensorboard --logdir=../runs/mnist_experiment

### Prédiction sur image externe  
python predict_external.py
