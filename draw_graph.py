import pandas as pd
import numpy as np
# %matplotlib inline
import matplotlib.pyplot as plt

# iris_training_acc = pd.read_csv('E:/backup/tensorboard/Warsaw/Proposed/Dense/iris/run-train_iris_1-fold_1-DenseNet-tag-Accuracy.csv')
# iris_training_acc.drop(['Wall time', 'Step'], axis=1, inplace=True)

gen_1fold_FT = pd.read_csv('log/ND/run-1-fold_loss_runs_nd_exp_1-tag-Gen_Fourior.csv')
gen_1fold_gan = pd.read_csv('log/ND/run-1-fold_loss_runs_nd_exp_1-tag-Gen_loss_G_GAN.csv')
gen_1fold_cycle = pd.read_csv('log/ND/run-1-fold_loss_runs_nd_exp_1-tag-Gen_loss_G_cycle.csv')
gen_1fold_id = pd.read_csv('log/ND/run-1-fold_loss_runs_nd_exp_1-tag-Gen_loss_G_ID.csv')

gen_2fold_FT = pd.read_csv('log/ND/run-2-fold_loss_runs_nd_exp_1-tag-Gen_Fourior.csv')
gen_2fold_gan = pd.read_csv('log/ND/run-2-fold_loss_runs_nd_exp_1-tag-Gen_loss_G_GAN.csv')
gen_2fold_cycle = pd.read_csv('log/ND/run-2-fold_loss_runs_nd_exp_1-tag-Gen_loss_G_cycle.csv')

# val_gen = gen_1fold_val
# val_disc = disc_1fold_val
# w_gen = (w_gen_1fold + w_gen_2fold) / 2
# w_disc = (w_disc_1fold + w_disc_2fold) / 2

gen_ft = gen_1fold_FT
gen_gan = gen_1fold_gan
gen_cycle = gen_1fold_cycle
gen_id = gen_1fold_id

# val_loss = val_acc['0']
gen_ft.drop(['Wall time', 'Step'], axis=1, inplace=True)
gen_gan.drop(['Wall time', 'Step'], axis=1, inplace=True)
gen_cycle.drop(['Wall time', 'Step'], axis=1, inplace=True)
gen_id.drop(['Wall time', 'Step'], axis=1, inplace=True)


TSBOARD_SMOOTHING = [0]

smooth = []
for ts_factor in TSBOARD_SMOOTHING:
    smooth.append(gen_ft.ewm(alpha=(1 - ts_factor)).mean())
    smooth.append(gen_gan.ewm(alpha=(1 - ts_factor)).mean())
    smooth.append(gen_cycle.ewm(alpha=(1 - ts_factor)).mean())
    smooth.append(gen_id.ewm(alpha=(1 - ts_factor)).mean())


# Create the same plot structure as before for these new files
# plt.plot(smooth[0], label='Training loss of gan', color='green')
# plt.plot(smooth[2], label='Training loss of cycle', color='red')
# plt.plot(smooth[3], label='Training loss of ID', color='blue')
fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.set_xlabel('Epoch', fontsize=16, labelpad=15)
ax1.set_ylabel('Cyclic and identity loss value', color='black')
ax1.plot(smooth[2], label='Training cyclic loss of generator', color='blue')
ax1.plot(smooth[1], label='Training gan loss of generator', color='red')
#
# # Create a second y-axis for the loss
ax2 = ax1.twinx()
ax2.set_ylabel('Identity loss value', color='black', labelpad=15)
# ax2.plot(smooth[0], label='Training loss of FT', color='green')
ax2.plot(smooth[3], label='Training identity loss of generator', color='orange')

# fig.legend(bbox_to_anchor=(0.3, 0.85), fontsize=15)
# ax1.legend()
# ax2.legend()
plt.show()
