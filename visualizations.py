import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from evaluate_models import MnistData

if __name__ == "__main__":
    
    imgs_dir_name = "images"
    os.makedirs(imgs_dir_name, exist_ok=True)
    
    data = MnistData(normalize=False)
        
    for db_name in ["mnist", "fashion"]:
        
        X_train, y_train = data(db_name, "train") 
        
        num_samples, num_features = X_train.shape
        num_classes = len(np.unique(y_train))
        num_side_length = int(np.sqrt(num_features))
        img_size = (num_side_length, num_side_length)
        
        # average densities
        print("Drawing chart of average density for {}...".format(db_name))
        avg_densities = []
        for i in range(num_classes):
            X_train_i = X_train[np.equal(y_train, i)]
            avg_X_train_i = np.mean(X_train_i, axis=0)
            avg_X_train_i.resize(img_size)
            avg_X_train_i = avg_X_train_i.astype(int)
            avg_densities.append(avg_X_train_i)
        
        fig, axes = plt.subplots(1, num_classes, figsize=(10, 1.5))
        for i in range(num_classes):
            axes[i].axis("off")
            axes[i].set_title(i)
            axes[i].imshow(avg_densities[i], cmap="gray")
        
        fig_name = "{}/avg_densities_{}.png".format(imgs_dir_name, db_name)
        fig.savefig(fig_name, format="png")
        
        # pairwise differences avg_densities
        print("Drawing chart of average density differences for {}...".format(db_name))
        avg_differences = [[None for _ in range(num_classes)] for _ in range(num_classes)]
        for i in range(num_classes):
            for j in range(num_classes):
                avg_differences[i][j] = avg_densities[i] - avg_densities[j]
        
        
        fig, axes = plt.subplots(num_classes+1, num_classes+1, figsize=(8, 8))
        axes[0, 0].axis("off")
        for i in range(num_classes):
            axes[i+1, 0].axis("off")
            axes[0, i+1].axis("off")
            axes[i+1, 0].text(0.5, 0.5, i)
            axes[0, i+1].text(0.5, 0.5, i)
        for i in range(num_classes):
            for j in range(num_classes):
                    axes[i+1, j+1].axis("off")
                    axes[i+1, j+1].imshow(avg_differences[i][j], vmin=-256, vmax=255, cmap="bwr")
                
        fig_name = "{}/avg_differences_{}.png".format(imgs_dir_name, db_name)
        fig.savefig(fig_name, format="png")
        
        # accuracies 1_px logistic pairwise
        print("Drawing chart of pairwise logistic regression for {}...".format(db_name))
        filename = "csv/accuracy_1px_pairwise_{}_out.csv".format(db_name)
        acc_out = np.loadtxt(filename)
        fig = plt.Figure(figsize=(9, 9))
        ax = fig.add_subplot()
        sns.heatmap(data=acc_out, vmin=0.5, vmax=1, cmap="RdYlGn", annot=True, ax=ax)
        
        fig_name = "{}/accuracy_1px_pairwise_{}.png".format(imgs_dir_name, db_name)
        fig.savefig(fig_name, format="png")
        
        
        
        
        